"""Run TDSTF on the held-out test set and plot predictive violins."""

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
from dataset import get_dataloader
from diff import TDSTF
from exe import calc_metrics
from evaluation import (
    TARGET_UNITS,
    calculate_predictive_metrics,
    extract_targets,
    legacy_nacrps_statistics,
    validate_forecasts,
)
from reproducibility import seed_everything, validate_split_seed


TARGET_DISPLAY_NAMES = {"O2 Saturation": "SpO₂"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infere no conjunto de teste e gera gráficos de previsão com violinos."
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


def collect_forecasts(model, data_loader, nsample, seed=2026):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    generations, targets, histories, infos = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            generation, samples_y, samples_x = model.evaluate(
                batch, nsample, generator=generator
            )
            generations.append(generation.detach().cpu())
            targets.append(samples_y.detach().cpu())
            histories.append(samples_x.detach().cpu())
            infos.append(batch["info"].detach().cpu())
    if not generations:
        raise RuntimeError("O conjunto de teste está vazio.")
    return tuple(torch.cat(items, dim=0) for items in (generations, targets, histories, infos))


def prediction_rows(generation, samples_y, info, variable_names, target_ids, means, stds):
    rows = []
    extracted = extract_targets(samples_y)
    generation = validate_forecasts(generation, extracted)
    gen, target, metadata = generation, samples_y.numpy(), info.numpy()
    valid = extracted.mask.numpy()
    feature_ids = extracted.feature_ids.numpy()
    actual_values = extracted.values.numpy()
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


def save_predictions_csv(rows, path):
    columns = ["sample_id", "signal", "minute", "actual", "predicted_median", "predicted_mean", "predicted_p025", "predicted_p975"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_example(sample_index, generation, samples_y, samples_x, info, variable_names,
                 target_ids, means, stds, output_path):
    extracted = extract_targets(samples_y[sample_index:sample_index + 1])
    gen, target, history = generation[sample_index].numpy(), samples_y[sample_index].numpy(), samples_x[sample_index].numpy()
    valid_targets = extracted.mask[0].numpy()
    target_values = extracted.values[0].numpy()
    target_features = extracted.feature_ids[0].numpy()
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

    axes[0].set_title(f"Previsão no conjunto de teste — internação {sample_id}")
    axes[-1].set_xlabel("Minuto relativo ao início da janela de 40 minutos")
    axes[-1].set_xlim(0, 40)
    legend = [
        Patch(facecolor="#2878b5", edgecolor="#2878b5", alpha=0.8, label="Real observado"),
        Patch(facecolor="#5aa1d6", edgecolor="#2878b5", alpha=0.55, label="Distribuição predita"),
        Patch(facecolor="#d62728", edgecolor="#d62728", label="Real a prever"),
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_signal_distribution(rows, signal_names, output_path):
    fig, axes = plt.subplots(
        len(signal_names), 1, figsize=(7.5, max(9, 2.7 * len(signal_names))), squeeze=False
    )
    for ax, signal in zip(axes[:, 0], signal_names):
        selected = [row for row in rows if row["signal"] == signal]
        if not selected:
            ax.set_visible(False)
            continue
        real = np.asarray([row["actual"] for row in selected], dtype=float)
        predicted = np.asarray([row["predicted_median"] for row in selected], dtype=float)
        parts = ax.violinplot([real, predicted], positions=[1, 2], widths=0.75,
                              showmeans=False, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], ["#d95f5f", "#5aa1d6"]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.6)
        parts["cmedians"].set_color("#222222")
        ax.set_xticks([1, 2], ["Real", "Mediana predita"])
        ax.set_ylabel(signal_label(signal))
        ax.set_title(f"Distribuição marginal de {TARGET_DISPLAY_NAMES.get(signal, signal)}")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Sinais reais e medianas previstas no conjunto de teste", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else checkpoint.parent / f"inference_n{args.nsample}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloader(
        "preprocess/data/dataset.pkl",
        "preprocess/data/var.pkl",
        config["diffusion"]["size"],
        batch_size=config["train"]["batch_size"],
        seed=args.data_seed,
    )

    model = TDSTF(config, device).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    generation, samples_y, samples_x, info = collect_forecasts(
        model, test_loader, args.nsample, seed=args.seed
    )

    NACRPS, _ = calc_metrics(1, generation, samples_y)
    predictive_metrics = calculate_predictive_metrics(
        generation, samples_y, variable_names, target_ids, means, stds,
        reference_scales=reference_scales, info=info,
    )
    metrics = {
        "checkpoint": str(checkpoint),
        "device": device,
        "nsample": args.nsample,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "test_samples": int(generation.shape[0]),
        "NACRPS": float(NACRPS),
        "NACRPS_statistics": legacy_nacrps_statistics(1, generation, samples_y),
        "NACRPS_definition": (
            "Legacy quantile score: mean over q=.05..95 step .05, normalized by "
            "sum(abs(valid standardized targets)); not directly comparable across tasks/cohorts."
        ),
        "evaluation_reference_scale": {
            key: reference.get(key) for key in ("version", "seed", "source")
        },
        "MSE_by_signal": {
            name: {
                "unit": values["unit"],
                "n_observations": values["n_observations"],
                "MSE_mean": values["metrics"].get("MSE_mean", {}),
            }
            for name, values in predictive_metrics["by_signal"].items()
        },
        "predictive_metrics": predictive_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    rows = prediction_rows(generation, samples_y, info, variable_names, target_ids, means, stds)
    save_predictions_csv(rows, output_dir / "predictions.csv")
    np.savez_compressed(
        output_dir / "posterior_samples.npz",
        generation=generation.numpy(), samples_y=samples_y.numpy(),
        samples_x=samples_x.numpy(), info=info.numpy(),
    )

    for sample_index in range(min(args.n_examples, len(generation))):
        sample_id = int(info[sample_index, 0])
        plot_example(
            sample_index, generation, samples_y, samples_x, info, variable_names,
            target_ids, means, stds, output_dir / f"forecast_example_{sample_id}.png",
        )
    signal_names = [variable_names[int(index)] for index in target_ids]
    plot_signal_distribution(rows, signal_names, output_dir / "signal_distribution_violin.png")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Resultados e figuras salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
