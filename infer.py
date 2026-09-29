"""Evaluate TDSTF on a selected patient split and save predictive diagnostics."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Patch
from baselines import evaluate_baselines
from dataset import get_dataloader
from diff import TDSTF
from evaluation import (
    TARGET_UNITS,
    calculate_predictive_metrics,
    extract_targets,
    legacy_nacrps_statistics,
    validate_forecasts,
)
from forecast_store import ForecastShardStore
from metrics_stream import PredictiveMetricsAccumulator
from reproducibility import seed_everything, validate_split_seed


TARGET_DISPLAY_NAMES = {"O2 Saturation": "SpO₂"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia um checkpoint em uma população selecionada e gera diagnósticos preditivos."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Caminho para model.pth, pasta do experimento ou nome da pasta dentro de save/.",
    )
    parser.add_argument("--device", default=None, help="Ex.: cuda:0 ou cpu. Padrão: CUDA se disponível.")
    parser.add_argument("--nsample", type=int, default=100, help="Amostras de difusão por previsão.")
    parser.add_argument("--n-examples", type=int, default=3, help="Número de internações a plotar em detalhe.")
    parser.add_argument("--output-dir", default=None, help="Pasta de saída (padrão: ao lado do checkpoint).")
    parser.add_argument("--seed", type=int, default=None, help="Semente das amostras (padrão: config/base.yaml).")
    parser.add_argument("--data-seed", type=int, default=None, help="Semente de seleção dos dados (padrão: seed do checkpoint).")
    parser.add_argument("--split", choices=("val_model", "calibration", "test"), default="val_model",
                        help="População de avaliação; test só é acessado quando selecionado explicitamente.")
    return parser.parse_args()


def resolve_checkpoint(value):
    supplied = Path(value).expanduser()
    if supplied.is_dir():
        candidates = [supplied / "model.pth"]
    else:
        candidates = [supplied, supplied / "model.pth", Path("save") / supplied / "model.pth"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Checkpoint não encontrado. Caminhos verificados: " + ", ".join(map(str, candidates)))


def signal_label(name):
    display_name = TARGET_DISPLAY_NAMES.get(name, name)
    unit = TARGET_UNITS.get(name, "")
    return f"{display_name} ({unit})" if unit else display_name


def unscale(values, feature_id, means, stds):
    scale = stds[feature_id] if stds[feature_id] != 0 else 1.0
    return values * scale + means[feature_id]


def iter_forecasts(model, data_loader, nsample, seed=2026):
    """Yield aligned forecast batches, immediately copied off the accelerator."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            generation, samples_y, samples_x = model.evaluate(
                batch, nsample, generator=generator
            )
            yield (
                generation.detach().cpu().numpy(),
                samples_y.detach().cpu().numpy(),
                samples_x.detach().cpu().numpy(),
                batch["info"].detach().cpu().numpy(),
            )


def prediction_rows(generation, samples_y, info, variable_names, target_ids, means, stds):
    rows = []
    extracted = extract_targets(samples_y)
    generation = validate_forecasts(generation, extracted)
    gen, target, metadata = np.asarray(generation), np.asarray(samples_y), np.asarray(info)
    valid = extracted.mask
    feature_ids = extracted.feature_ids
    actual_values = extracted.values
    for sample_index in range(len(gen)):
        sample_id = int(metadata[sample_index, 0])
        minutes = target[sample_index, 1]
        actuals = actual_values[sample_index]
        for feature_id in target_ids:
            feature_id = int(feature_id)
            for position in np.flatnonzero(valid[sample_index] & (feature_ids[sample_index] == feature_id)):
                draws = unscale(gen[sample_index, position], feature_id, means, stds)
                actual = float(unscale(actuals[position], feature_id, means, stds))
                rows.append({
                    "sample_id": sample_id,
                    "signal": variable_names[feature_id],
                    "minute": float(minutes[position]),
                    "actual": actual,
                    "predicted_median": float(np.median(draws)),
                    "predicted_mean": float(np.mean(draws)),
                    "predicted_p025": float(np.quantile(draws, 0.025)),
                    "predicted_p975": float(np.quantile(draws, 0.975)),
                })
    return rows


def mse_by_signal(generation, samples_y, variable_names, target_ids, means, stds,
                  reference_scales=None, info=None):
    """Compute mean-forecast MSE by signal, preserving clinical units."""
    detailed = calculate_predictive_metrics(
        generation, samples_y, variable_names, target_ids, means, stds,
        reference_scales=reference_scales, info=info,
    )
    return {
        name: {
            "unit": values["unit"],
            "n_observations": values["n_observations"],
            "MSE_mean": values["metrics"].get("MSE_mean", {}),
        }
        for name, values in detailed["by_signal"].items()
    }


def plot_example(sample_index, generation, samples_y, samples_x, info, variable_names,
                 target_ids, means, stds, output_path):
    generation, samples_y, samples_x, info = map(np.asarray, (generation, samples_y, samples_x, info))
    extracted = extract_targets(samples_y[sample_index:sample_index + 1])
    gen, target, history = generation[sample_index], samples_y[sample_index], samples_x[sample_index]
    valid_targets = extracted.mask[0]
    target_values = extracted.values[0]
    target_features = extracted.feature_ids[0]
    sample_id = int(info[sample_index, 0])
    fig, axes = plt.subplots(
        len(target_ids), 1, figsize=(10, max(9, 2.7 * len(target_ids))),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, feature_id in zip(axes, target_ids):
        feature_id = int(feature_id)
        history_positions = np.flatnonzero((history[3] > 0) & (history[0].astype(np.int64) == feature_id))
        if len(history_positions):
            ax.scatter(
                history[1, history_positions],
                unscale(history[2, history_positions], feature_id, means, stds),
                s=22, color="#2878b5", label="Real observado", zorder=4,
            )

        positions = np.flatnonzero(valid_targets & (target_features == feature_id))
        if len(positions):
            times = target[1, positions]
            actual = unscale(target_values[positions], feature_id, means, stds)
            draws = np.asarray([unscale(gen[position], feature_id, means, stds) for position in positions])
            parts = ax.violinplot(
                [draws[i] for i in range(len(positions))], positions=times, widths=0.8,
                showmeans=False, showmedians=True, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor("#5aa1d6")
                body.set_edgecolor("#2878b5")
                body.set_alpha(0.55)
            parts["cmedians"].set_color("#173f5f")
            parts["cmedians"].set_linewidth(1.2)
            low, high = np.quantile(draws, [0.025, 0.975], axis=1)
            ax.vlines(times, low, high, color="#173f5f", linewidth=1.1, zorder=3)
            ax.scatter(times, actual, s=28, color="#d62728", label="Real a prever", zorder=5)
        else:
            ax.text(0.5, 0.5, "Sem observações alvo deste sinal", ha="center", va="center", transform=ax.transAxes)

        ax.axvline(30, color="#777777", linestyle="--", linewidth=1, alpha=0.75)
        ax.set_ylabel(signal_label(variable_names[feature_id]))
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_title(f"Trajetórias preditivas — internação {sample_id}")
    axes[-1].set_xlabel("Minuto relativo ao início da janela de 40 minutos")
    axes[-1].set_xlim(0, 40)
    legend = [
        Patch(facecolor="#2878b5", edgecolor="#2878b5", alpha=0.8, label="Real observado"),
        Patch(facecolor="#5aa1d6", edgecolor="#2878b5", alpha=0.55, label="Amostras; linha = mediana"),
        Patch(facecolor="#d62728", edgecolor="#d62728", label="Real a prever"),
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_diagnostics(metrics, signal_names, output_dir):
    for signal in signal_names:
        detail = metrics["by_signal"].get(signal, {})
        diagnostics = detail.get("diagnostics", {})
        quantile_rows = diagnostics.get("quantiles_by_query_minute", {})
        fig, axes = plt.subplots(3, 1, figsize=(9, 10))

        if quantile_rows:
            query_minutes = sorted(float(value) for value in quantile_rows)
            observed = np.asarray([
                quantile_rows[str(value)]["observed_sum"] / quantile_rows[str(value)]["count"]
                for value in query_minutes
            ])
            axes[0].plot(query_minutes, observed, "ko-", label="Real (média por consulta)")
            quantile_levels = ("0.05", "0.25", "0.5", "0.75", "0.95")
            for level in quantile_levels:
                forecast = np.asarray([
                    quantile_rows[str(value)]["predicted_quantile_sums"][level] / quantile_rows[str(value)]["count"]
                    for value in query_minutes
                ])
                axes[0].plot(query_minutes, forecast, label=f"Amostras q={level}")
            axes[0].set_ylabel(signal_label(signal))
            axes[0].set_xlabel("Minuto da consulta")
            axes[0].legend(ncol=3, fontsize="small")
            axes[0].grid(alpha=0.2)
        else:
            axes[0].text(0.5, 0.5, "Sem consultas válidas", ha="center", va="center", transform=axes[0].transAxes)

        pit = diagnostics.get("pit", {})
        pit_counts = np.asarray(pit.get("histogram", []), dtype=float)
        if pit_counts.sum():
            axes[1].bar(np.linspace(0.05, 0.95, len(pit_counts)), pit_counts / pit_counts.sum(), width=0.08)
        axes[1].axhline(1 / 10, color="#d62728", linestyle="--", linewidth=1)
        axes[1].set(xlim=(0, 1), xlabel="PIT (mid-rank)", ylabel="Frequência", title="Calibração marginal — PIT/ranks")
        axes[1].grid(axis="y", alpha=0.2)

        axes[2].plot([0, 1], [0, 1], color="#777777", linestyle="--", label="Ideal")
        for tail, event in diagnostics.get("event_reliability", {}).items():
            bins = [value for value in event["bins"] if value["count"]]
            if bins:
                predicted = [value["predicted_probability_sum"] / value["count"] for value in bins]
                observed = [value["observed_event_sum"] / value["count"] for value in bins]
                axes[2].plot(predicted, observed, "o-", label=f"Cauda {tail}; limiar {event['threshold']:g}")
        axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Probabilidade prevista do evento", ylabel="Frequência observada", title="Confiabilidade dos eventos")
        axes[2].legend(fontsize="small")
        axes[2].grid(alpha=0.2)
        fig.suptitle(f"{TARGET_DISPLAY_NAMES.get(signal, signal)} — amostras, quantis e calibração")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"diagnostics_{signal.replace(' ', '_')}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    if args.nsample < 1:
        raise ValueError("--nsample deve ser pelo menos 1.")
    checkpoint = resolve_checkpoint(args.checkpoint)
    with Path("config/base.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    args.seed = config.get("seed", 2026) if args.seed is None else args.seed
    if args.seed < 0:
        raise ValueError("--seed deve ser um inteiro não negativo.")
    seed_everything(args.seed)

    run_metadata_path = checkpoint.parent / "run_metadata.json"
    if run_metadata_path.is_file():
        with run_metadata_path.open("r", encoding="utf-8") as metadata_file:
            run_metadata = json.load(metadata_file)
        default_data_seed = run_metadata.get("data_seed", run_metadata.get("seed", config.get("seed", 2026)))
    else:
        default_data_seed = config.get("seed", 2026)
    args.data_seed = default_data_seed if args.data_seed is None else args.data_seed
    if args.data_seed < 0:
        raise ValueError("--data-seed deve ser um inteiro não negativo.")
    validate_split_seed("preprocess/data/splits.pkl", args.data_seed)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas não está disponível. Use --device cpu.")
    with Path("preprocess/data/var.pkl").open("rb") as file:
        variable_names, target_ids = pickle.load(file)
    with Path("preprocess/data/mean_std.pkl").open("rb") as file:
        means, stds = pickle.load(file)
    reference_path = Path("preprocess/data/evaluation_reference_scale.pkl")
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Frozen evaluation scales are missing. Run preprocessing step_4.py once to create "
            "preprocess/data/evaluation_reference_scale.pkl; keep that file unchanged when updating the normalizer."
        )
    with reference_path.open("rb") as file:
        reference = pickle.load(file)
    if reference.get("variable_names") != list(variable_names) or not np.array_equal(reference.get("target_ids"), target_ids):
        raise ValueError("Frozen evaluation scales do not match var.pkl")
    reference_scales = np.asarray(reference["scales"], dtype=float)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else checkpoint.parent / f"inference_{args.split}_n{args.nsample}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_model_loader, calibration_loader, test_loader = get_dataloader(
        "preprocess/data/dataset.pkl",
        "preprocess/data/var.pkl",
        config["diffusion"]["size"],
        batch_size=config["train"]["batch_size"],
        seed=args.data_seed,
        include_test=args.split == "test",
    )
    loaders = {
        "val_model": val_model_loader,
        "calibration": calibration_loader,
        "test": test_loader,
    }
    data_loader = loaders[args.split]
    if data_loader is None:
        raise ValueError(f"Split '{args.split}' is unavailable; regenerate data with preprocess/step_4.py")

    model = TDSTF(config, device).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    shard_store = ForecastShardStore(
        output_dir / "posterior_samples",
        {
            "split": args.split,
            "population": f"{args.split} patient partition",
            "source": str(checkpoint),
            "nsample": args.nsample,
            "seed": args.seed,
            "evaluation_reference_scale": {key: reference.get(key) for key in ("version", "seed", "source")},
        },
    )
    metric_accumulator = PredictiveMetricsAccumulator()
    nacrps_numerator = nacrps_denominator = 0.0
    nacrps_count = valid_count = sample_count = batch_count = 0
    example_count = 0
    tail_thresholds = config.get("evaluation", {}).get("tail_thresholds", {})
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as predictions_file:
        columns = ["sample_id", "signal", "minute", "actual", "predicted_median", "predicted_mean", "predicted_p025", "predicted_p975"]
        writer = csv.DictWriter(predictions_file, fieldnames=columns)
        writer.writeheader()
        for generation, samples_y, samples_x, info in iter_forecasts(
            model, data_loader, args.nsample, seed=args.seed
        ):
            batch_count += 1
            sample_count += generation.shape[0]
            shard_store.add(generation, samples_y, samples_x, info)
            target = extract_targets(samples_y)
            valid_count += int(target.mask.sum())
            nacrps = legacy_nacrps_statistics(args.split == "test", generation, samples_y)
            if nacrps["numerator"] is not None:
                nacrps_numerator += nacrps["numerator"]
                nacrps_denominator += nacrps["denominator"]
                nacrps_count += nacrps["count"]
            report = calculate_predictive_metrics(
                generation, samples_y, variable_names, target_ids, means, stds,
                reference_scales=reference_scales, info=info,
                tail_thresholds=tail_thresholds, include_patient_statistics=True,
            )
            metric_accumulator.add(report)
            writer.writerows(prediction_rows(
                generation, samples_y, info, variable_names, target_ids, means, stds
            ))
            while example_count < min(args.n_examples, sample_count):
                local_index = example_count - (sample_count - generation.shape[0])
                sample_id = int(info[local_index, 0])
                plot_example(
                    local_index, generation, samples_y, samples_x, info, variable_names,
                    target_ids, means, stds,
                    output_dir / f"forecast_example_{example_count}_{sample_id}.png",
                )
                example_count += 1

    if batch_count == 0:
        raise ValueError(f"Evaluation split '{args.split}' contains no batches")
    predictive_metrics = metric_accumulator.finalize()
    NACRPS = nacrps_numerator / nacrps_denominator if nacrps_denominator else None
    nacrps_stats = {
        "numerator": nacrps_numerator if nacrps_count else None,
        "denominator": nacrps_denominator,
        "count": nacrps_count,
        "value": NACRPS,
    }
    if NACRPS is None:
        nacrps_stats["reason"] = "no_valid_observations" if not nacrps_count else "zero_absolute_target_sum"
    metrics = {
        "split": args.split,
        "population": f"{args.split} patient partition",
        "source": str(checkpoint),
        "measurement_source": "MIMIC-IV preprocessed minute-level signals; event-level provenance is not retained by the current aggregation",
        "device": device,
        "nsample": args.nsample,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "test_samples": sample_count,
        "valid_targets": valid_count,
        "NACRPS": nacrps_stats,
        "NACRPS_definition": "Legacy quantile formula, normalized by sum(abs(valid standardized targets)); not comparable across tasks/cohorts.",
        "evaluation_reference_scale": {key: reference.get(key) for key in ("version", "seed", "source")},
        "MSE_by_signal": {
            name: {"unit": values["unit"], "n_observations": values["n_observations"],
                   "MSE_mean": values["metrics"].get("MSE_mean", {})}
            for name, values in predictive_metrics["by_signal"].items()
        },
        "predictive_metrics": predictive_metrics,
        "shards": str(shard_store.directory),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False, allow_nan=False)
    evaluate_baselines(
        train_loader, data_loader, variable_names, target_ids, means, stds,
        reference_scales, args.split, output_dir / "baseline_metrics.json",
        tail_thresholds=tail_thresholds,
    )
    signal_names = [variable_names[int(index)] for index in target_ids]
    plot_forecast_diagnostics(predictive_metrics, signal_names, output_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Resultados e shards salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
