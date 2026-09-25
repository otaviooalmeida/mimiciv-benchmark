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
from torch.utils.data import DataLoader

from dataset import get_dataloader
from diff import TDSTF
from exe import calc_metrics
from preprocess.windowing import HISTORY_MINUTES, FORECAST_MINUTES, WINDOW_MINUTES


TARGET_UNITS = {"HR": "bpm", "SBP": "mmHg", "DBP": "mmHg", "Temperature": "°C", "O2 Saturation": "%"}
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
    parser.add_argument("--n-examples", type=int, default=3, help="Número de janelas a plotar em detalhe.")
    parser.add_argument("--output-dir", default=None, help="Pasta de saída (padrão: ao lado do checkpoint).")
    parser.add_argument("--seed", type=int, default=2026, help="Semente para reproduzir as amostras.")
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
        history_positions = np.flatnonzero((history[3] > 0) & (history[0].astype(np.int64) == feature_id))
        if len(history_positions):
            ax.scatter(
                history[1, history_positions],
                unscale(history[2, history_positions], feature_id, means, stds),
                s=22, color="#2878b5", label="Real observado", zorder=4,
            )

        positions = np.flatnonzero((target[3] > 0) & (target[0].astype(np.int64) == feature_id))
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
    )
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
        "test_samples": int(generation.shape[0]),
        "history_minutes": HISTORY_MINUTES,
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

    for sample_index in range(min(args.n_examples, len(generation))):
        sample_id = int(info[sample_index, 0])
        window_start = int(info[sample_index, 3])
        plot_example(
            sample_index, generation, samples_y, samples_x, info, variable_names,
            target_ids, means, stds, output_dir / f"forecast_example_{sample_id}_window_{window_start}.png",
        )
    signal_names = [variable_names[int(index)] for index in target_ids]
    plot_signal_distribution(rows, signal_names, output_dir / "signal_distribution_violin.png")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Resultados e figuras salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
