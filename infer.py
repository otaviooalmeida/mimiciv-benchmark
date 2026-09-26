"""Run TDSTF and inspect marginal distributions, trajectories and event risks."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from dataset import get_dataloader
from diff import TDSTF
from exe import calc_metrics
from preprocess.windowing import HISTORY_MINUTES, FORECAST_MINUTES, WINDOW_MINUTES


TARGET_UNITS = {"HR": "bpm", "SBP": "mmHg", "RR": "brpm", "Temperature": "°C", "O2 Saturation": "%"}
TARGET_DISPLAY_NAMES = {"O2 Saturation": "SpO₂"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Infere no teste e gera violinos, trajetórias e probabilidades de eventos."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Caminho para model.pth, pasta do experimento ou nome da pasta dentro de save/.",
    )
    parser.add_argument("--device", default=None, help="Ex.: cuda:0 ou cpu. Padrão: CUDA se disponível.")
    parser.add_argument("--nsample", type=int, default=100, help="Amostras de difusão por previsão.")
    parser.add_argument("--n-examples", type=int, default=3, help="Número de janelas a plotar em detalhe.")
    parser.add_argument("--n-trajectories", type=int, default=20,
                        help="Trajetórias completas por gráfico, limitadas por --nsample (padrão: 20).")
    parser.add_argument("--event-config", type=Path, default=None,
                        help="YAML com min_change e max_gap_minutes por sinal, em unidades originais.")
    parser.add_argument("--output-dir", default=None, help="Pasta de saída (padrão: ao lado do checkpoint).")
    parser.add_argument("--seed", type=int, default=2026, help="Semente para reproduzir as amostras.")
    args = parser.parse_args(argv)
    if args.nsample < 1 or args.n_trajectories < 1 or args.n_examples < 0:
        parser.error("--nsample e --n-trajectories devem ser positivos; --n-examples deve ser >= 0.")
    return args


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


def signal_positions(sample, feature_id):
    """Indices of observed triplets for one signal, ordered by time."""
    positions = np.flatnonzero((sample[3] > 0) & (sample[0] == feature_id))
    return positions[np.argsort(sample[1, positions], kind="stable")]


def load_event_rules(path):
    """Require explicit, positive thresholds; do not fit them on held-out data."""
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as file:
        rules = yaml.safe_load(file)
    if not isinstance(rules, dict) or not rules:
        raise ValueError("A configuração de eventos deve conter regras por sinal.")
    for signal, rule in rules.items():
        if signal not in TARGET_UNITS:
            raise ValueError(f"Sinal desconhecido na configuração de eventos: {signal}")
        if not isinstance(rule, dict) or set(rule) != {"min_change", "max_gap_minutes"}:
            raise ValueError(f"{signal}: especifique min_change e max_gap_minutes.")
        for key, value in rule.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{signal}.{key} deve ser um número finito e positivo.")
    return rules


def summarize_events(times, actual, draws, rule, last_observation=None):
    """Count trajectories with a rise/drop on eligible consecutive same-signal pairs.

    Values are in original units; draws has shape [observations, trajectories].
    These are events on the queried observation grid, not between measurements.
    """
    times, actual, draws = np.asarray(times), np.asarray(actual), np.asarray(draws)
    if (times.ndim != 1 or actual.shape != times.shape or draws.ndim != 2
            or draws.shape[0] != len(times) or draws.shape[1] < 1):
        raise ValueError("Dimensões incompatíveis para calcular eventos.")
    if not all(np.isfinite(values).all() for values in (times, actual, draws)):
        raise ValueError("Observações e previsões válidas devem ser finitas.")
    order = np.argsort(times, kind="stable")
    times, actual, draws = times[order], actual[order], draws[order]
    if len(times) and last_observation is not None:
        last_time, last_value = last_observation
        if not np.isfinite([last_time, last_value]).all() or last_time >= times[0]:
            raise ValueError("A âncora do histórico deve ser finita e anterior ao primeiro alvo.")
        times = np.concatenate(([last_time], times))
        actual = np.concatenate(([last_value], actual))
        draws = np.concatenate((np.full((1, draws.shape[1]), last_value), draws), axis=0)
    gaps = np.diff(times)
    eligible = (gaps > 0) & (gaps <= rule["max_gap_minutes"])
    summary = {
        "eligible_pairs": int(eligible.sum()), "n_draws": int(draws.shape[1]),
        "p_rise": None, "p_drop": None, "observed_rise": None, "observed_drop": None,
    }
    if not eligible.any():
        return summary
    changes = np.diff(draws, axis=0)[eligible]
    actual_changes = np.diff(actual)[eligible]
    threshold = rule["min_change"]
    summary.update(
        p_rise=float(np.any(changes >= threshold, axis=0).mean()),
        p_drop=float(np.any(changes <= -threshold, axis=0).mean()),
        observed_rise=bool(np.any(actual_changes >= threshold)),
        observed_drop=bool(np.any(actual_changes <= -threshold)),
    )
    return summary


def event_rows(generation, samples_y, samples_x, info, variable_names, target_ids,
               means, stds, rules):
    rows = []
    gen, targets, histories = generation.numpy(), samples_y.numpy(), samples_x.numpy()
    for i, (target, history) in enumerate(zip(targets, histories)):
        for feature_id in target_ids:
            feature_id = int(feature_id)
            signal = variable_names[feature_id]
            if signal not in rules:
                continue
            positions = signal_positions(target, feature_id)
            history_positions = signal_positions(history, feature_id)
            last_observation = None
            if len(history_positions):
                last = history_positions[-1]
                last_observation = (history[1, last], unscale(history[2, last], feature_id, means, stds))
            summary = summarize_events(
                target[1, positions], unscale(target[2, positions], feature_id, means, stds),
                unscale(gen[i, positions], feature_id, means, stds), rules[signal], last_observation,
            )
            rows.append({
                "sample_id": int(info[i, 0]), "window_start": int(info[i, 3]),
                "signal": signal, "unit": TARGET_UNITS[signal], **rules[signal], **summary,
            })
    return rows


def save_events_csv(rows, path):
    columns = ["sample_id", "window_start", "signal", "unit", "min_change", "max_gap_minutes",
               "eligible_pairs", "n_draws", "p_rise", "p_drop", "observed_rise", "observed_drop"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def collect_forecasts(model, data_loader, nsample):
    generations, targets, histories, infos = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            generation, samples_y, samples_x = model.evaluate(batch, nsample)
            generations.append(generation.detach().cpu())
            targets.append(samples_y.detach().cpu())
            histories.append(samples_x.detach().cpu())
            infos.append(batch["info"].detach().cpu())
    if not generations:
        raise RuntimeError("O conjunto de teste está vazio.")
    return tuple(torch.cat(items, dim=0) for items in (generations, targets, histories, infos))


def prediction_rows(generation, samples_y, info, variable_names, target_ids, means, stds):
    rows = []
    gen, target, metadata = generation.numpy(), samples_y.numpy(), info.numpy()
    for sample_index in range(len(gen)):
        sample_id = int(metadata[sample_index, 0])
        valid = target[sample_index, 3] > 0
        feature_ids = target[sample_index, 0].astype(np.int64)
        minutes, actuals = target[sample_index, 1], target[sample_index, 2]
        for feature_id in target_ids:
            feature_id = int(feature_id)
            for position in np.flatnonzero(valid & (feature_ids == feature_id)):
                draws = unscale(gen[sample_index, position], feature_id, means, stds)
                actual = float(unscale(actuals[position], feature_id, means, stds))
                rows.append({
                    "sample_id": sample_id,
                    "window_start": int(metadata[sample_index, 3]),
                    "signal": variable_names[feature_id],
                    "minute": float(minutes[position]),
                    "actual": actual,
                    "predicted_median": float(np.median(draws)),
                    "predicted_mean": float(np.mean(draws)),
                    "predicted_p025": float(np.quantile(draws, 0.025)),
                    "predicted_p975": float(np.quantile(draws, 0.975)),
                })
    return rows


def save_predictions_csv(rows, path):
    columns = ["sample_id", "window_start", "signal", "minute", "actual", "predicted_median", "predicted_mean", "predicted_p025", "predicted_p975"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_example(sample_index, generation, samples_y, samples_x, info, variable_names,
                 target_ids, means, stds, output_path):
    gen, target, history = generation[sample_index].numpy(), samples_y[sample_index].numpy(), samples_x[sample_index].numpy()
    sample_id = int(info[sample_index, 0])
    window_start = int(info[sample_index, 3])
    fig, axes = plt.subplots(
        len(target_ids), 1, figsize=(10, max(9, 2.7 * len(target_ids))),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, feature_id in zip(axes, target_ids):
        feature_id = int(feature_id)
        history_positions = signal_positions(history, feature_id)
        if len(history_positions):
            ax.scatter(
                history[1, history_positions],
                unscale(history[2, history_positions], feature_id, means, stds),
                s=22, color="#2878b5", label="Real observado", zorder=4,
            )

        positions = signal_positions(target, feature_id)
        if len(positions):
            times = target[1, positions]
            actual = unscale(target[2, positions], feature_id, means, stds)
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

        ax.axvline(HISTORY_MINUTES, color="#777777", linestyle="--", linewidth=1, alpha=0.75)
        ax.set_ylabel(signal_label(variable_names[feature_id]))
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_title(f"Previsão no conjunto de teste — internação {sample_id}, janela no minuto {window_start}")
    axes[-1].set_xlabel(f"Minuto relativo ao início da janela de {WINDOW_MINUTES} minutos")
    axes[-1].set_xlim(0, WINDOW_MINUTES)
    legend = [
        Patch(facecolor="#2878b5", edgecolor="#2878b5", alpha=0.8, label="Real observado"),
        Patch(facecolor="#5aa1d6", edgecolor="#2878b5", alpha=0.55, label="Distribuição predita"),
        Patch(facecolor="#d62728", edgecolor="#d62728", label="Real a prever"),
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(sample_index, generation, samples_y, samples_x, info, variable_names,
                      target_ids, means, stds, output_path, n_trajectories=20, seed=2026,
                      events=None):
    """Plot whole draws, using the same draw indices for every time and signal."""
    gen = generation[sample_index].numpy()
    target, history = samples_y[sample_index].numpy(), samples_x[sample_index].numpy()
    sample_id, window_start = int(info[sample_index, 0]), int(info[sample_index, 3])
    # Local RNG: visual subsampling must not alter diffusion or depend on ground truth.
    indices = np.sort(np.random.default_rng(seed).choice(
        gen.shape[-1], size=min(n_trajectories, gen.shape[-1]), replace=False,
    ))
    fig, axes = plt.subplots(len(target_ids), 1, figsize=(11, max(9, 3 * len(target_ids))),
                             sharex=True, squeeze=False)
    for ax, feature_id in zip(axes[:, 0], target_ids):
        feature_id = int(feature_id)
        signal = variable_names[feature_id]
        history_positions = signal_positions(history, feature_id)
        ax.scatter(history[1, history_positions],
                   unscale(history[2, history_positions], feature_id, means, stds),
                   s=16, color="#2878b5", zorder=4)
        positions = signal_positions(target, feature_id)
        if len(positions):
            times = target[1, positions]
            draws = unscale(gen[positions], feature_id, means, stds)
            actual = unscale(target[2, positions], feature_id, means, stds)
            low, high = np.quantile(draws, [0.025, 0.975], axis=1)
            ax.fill_between(times, low, high, color="#5aa1d6", alpha=0.15)
            ax.vlines(times, low, high, color="#5aa1d6", alpha=0.3, linewidth=0.8)
            for index in indices:
                ax.plot(times, draws[:, index], color="#5aa1d6", alpha=0.3,
                        linewidth=0.8, marker=".", markersize=2, gid=f"trajectory-{index}")
            ax.plot(times, np.median(draws, axis=1), color="#173f5f", linewidth=2,
                    marker=".", label="Mediana por instante", zorder=5)
            ax.plot(times, actual, color="#d62728", linewidth=1.5, marker="o",
                    markersize=3, label="Real a prever", zorder=6)
        else:
            ax.text(0.5, 0.5, "Sem observações alvo deste sinal", ha="center", transform=ax.transAxes)
        event = (events or {}).get(signal)
        if event is not None:
            if event["eligible_pairs"]:
                text = (f"P(subida)={event['p_rise']:.1%} | P(queda)={event['p_drop']:.1%}"
                        f" | {event['eligible_pairs']} pares, {event['n_draws']} trajetórias")
            else:
                text = "P(evento): indisponível — sem pares elegíveis"
            text += f"\nMudança ≥ {event['min_change']:g} {event['unit']} em ≤ {event['max_gap_minutes']:g} min"
            ax.text(0.01, 0.98, text, va="top", fontsize=8, transform=ax.transAxes,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"}, zorder=10)
        ax.axvline(HISTORY_MINUTES, color="#777777", linestyle="--", linewidth=1)
        ax.set_ylabel(signal_label(signal))
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_title(f"Trajetórias — internação {sample_id}, janela no minuto {window_start}")
    axes[-1, 0].set_xlabel("Minuto relativo ao início da janela; linhas apenas conectam observações")
    axes[-1, 0].set_xlim(0, WINDOW_MINUTES)
    legend = [
        Line2D([], [], color="#2878b5", marker="o", linestyle="none", label="Histórico observado"),
        Line2D([], [], color="#5aa1d6", alpha=0.6, label=f"{len(indices)} trajetórias completas"),
        Line2D([], [], color="#173f5f", linewidth=2, label="Mediana por instante"),
        Line2D([], [], color="#d62728", marker="o", label="Real a prever"),
        Patch(facecolor="#5aa1d6", alpha=0.15, label="Intervalo marginal 95%"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def predictive_distribution_samples(generation, samples_y, variable_names, target_ids,
                                    means, stds, seed=2026, max_points=10000):
    """Bounded uniform samples of (observed target position, diffusion draw) pairs."""
    gen, target = generation.numpy(), samples_y.numpy()
    rng = np.random.default_rng(seed)
    result = {}
    for feature_id in target_ids:
        feature_id = int(feature_id)
        batch, position = np.nonzero((target[:, 3] > 0) & (target[:, 0] == feature_id))
        total = len(batch) * gen.shape[-1]
        flat = rng.choice(total, size=min(total, max_points), replace=False)
        observation, draw = np.divmod(flat, gen.shape[-1])
        values = gen[batch[observation], position[observation], draw]
        result[variable_names[feature_id]] = unscale(values, feature_id, means, stds)
    return result


def plot_signal_distribution(rows, signal_names, output_path, predictive_samples=None,
                             seed=2026, max_points=10000):
    rng = np.random.default_rng(seed)
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
        values, labels = [real, predicted], ["Real", "Medianas previstas"]
        if predictive_samples is not None and len(predictive_samples.get(signal, [])):
            values.append(predictive_samples[signal])
            labels.append("Amostras preditivas")
        # Bound KDE work without selecting values according to their magnitude.
        values = [v[rng.choice(len(v), size=min(len(v), max_points), replace=False)] for v in values]
        parts = ax.violinplot(values, positions=np.arange(1, len(values) + 1), widths=0.75,
                              showmeans=False, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], ["#d95f5f", "#5aa1d6", "#65a765"]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.6)
        parts["cmedians"].set_color("#222222")
        ax.set_xticks(np.arange(1, len(labels) + 1), labels)
        ax.set_ylabel(signal_label(signal))
        ax.set_title(f"Distribuição marginal de {TARGET_DISPLAY_NAMES.get(signal, signal)}")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Distribuições marginais: valores, resumos pontuais e amostras\n"
                 "Semelhança marginal não comprova acerto temporal ou calibração condicional", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    event_rules = load_event_rules(args.event_config)
    checkpoint = resolve_checkpoint(args.checkpoint)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas não está disponível. Use --device cpu.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with Path("config/base.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with Path("preprocess/data/var.pkl").open("rb") as file:
        variable_names, target_ids = pickle.load(file)
    with Path("preprocess/data/mean_std.pkl").open("rb") as file:
        means, stds = pickle.load(file)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else checkpoint.parent / f"inference_n{args.nsample}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader_shuffled = get_dataloader(
        "preprocess/data/dataset.pkl", "preprocess/data/var.pkl", config["diffusion"]["size"],
        batch_size=config["train"]["batch_size"],
        recent_per_target=config["diffusion"].get("recent_per_target", 3),
    )
    # Disk-backed datasets pin their normalization snapshot. Do not mix it with
    # a mean_std.pkl replaced by another preprocessing run.
    manifest = getattr(test_loader_shuffled.dataset, 'manifest', None)
    if manifest is not None:
        means, stds = manifest['means'], manifest['stds']
    test_loader = DataLoader(
        test_loader_shuffled.dataset,
        batch_size=test_loader_shuffled.batch_size,
        shuffle=False,
        num_workers=test_loader_shuffled.num_workers,
        pin_memory=test_loader_shuffled.pin_memory,
    )

    model = TDSTF(config, device).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    generation, samples_y, samples_x, info = collect_forecasts(model, test_loader, args.nsample)

    sacrps, mse = calc_metrics(1, generation, samples_y)
    metrics = {
        "checkpoint": str(checkpoint),
        "device": device,
        "nsample": args.nsample,
        "seed": args.seed,
        "n_trajectories_plotted": min(args.n_trajectories, args.nsample),
        "event_rules": event_rules,
        "event_scope": "Consecutive observed same-signal pairs; latest selected history observation may anchor the first target.",
        "distribution_plot_max_points": 10000,
        "test_samples": int(generation.shape[0]),
        "history_minutes": HISTORY_MINUTES,
        "history_selection": {
            "policy": "recent_targets_balanced_context",
            "size": config["diffusion"]["size"],
            "recent_per_target": config["diffusion"].get("recent_per_target", 3),
        },
        "forecast_minutes": FORECAST_MINUTES,
        "SACRPS": float(sacrps),
        "MSE": float(mse.item() if torch.is_tensor(mse) else mse),
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

    events_by_window = {}
    if event_rules:
        events = event_rows(generation, samples_y, samples_x, info, variable_names,
                            target_ids, means, stds, event_rules)
        save_events_csv(events, output_dir / "event_probabilities.csv")
        for row in events:
            events_by_window.setdefault((row["sample_id"], row["window_start"]), {})[row["signal"]] = row

    for sample_index in range(min(args.n_examples, len(generation))):
        sample_id = int(info[sample_index, 0])
        window_start = int(info[sample_index, 3])
        plot_example(
            sample_index, generation, samples_y, samples_x, info, variable_names,
            target_ids, means, stds, output_dir / f"forecast_example_{sample_id}_window_{window_start}.png",
        )
        plot_trajectories(
            sample_index, generation, samples_y, samples_x, info, variable_names,
            target_ids, means, stds, output_dir / f"forecast_trajectories_{sample_id}_window_{window_start}.png",
            n_trajectories=args.n_trajectories, seed=args.seed,
            events=events_by_window.get((sample_id, window_start)),
        )
    signal_names = [variable_names[int(index)] for index in target_ids]
    predictive_samples = predictive_distribution_samples(
        generation, samples_y, variable_names, target_ids, means, stds, seed=args.seed,
    )
    plot_signal_distribution(rows, signal_names, output_dir / "signal_distribution_violin.png",
                             predictive_samples=predictive_samples, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Resultados e figuras salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
