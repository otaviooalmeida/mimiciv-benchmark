"""Online train-only normalization and sharded legacy dataset generation."""

import csv
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from reproducibility import split_subjects, split_validation_subjects
except ImportError:  # pragma: no cover - project root is normally on sys.path
    from ..reproducibility import split_subjects, split_validation_subjects


INDEX_COLUMNS = ["ts_ind", "sub_id", "x_len", "y_len", "sample_file"]


class OnlineFeatureStats:
    """Chan/Welford population statistics with explicit non-finite tracking."""

    def __init__(self, feature_count):
        self.count = np.zeros(feature_count, dtype=np.int64)
        self.mean = np.zeros(feature_count, dtype=np.float64)
        self.m2 = np.zeros(feature_count, dtype=np.float64)
        self.invalid = np.zeros(feature_count, dtype=bool)

    def update(self, feature_ids, values):
        feature_ids = np.asarray(feature_ids, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        for feature_id in np.unique(feature_ids):
            selected = values[feature_ids == feature_id]
            if not np.isfinite(selected).all():
                self.invalid[feature_id] = True
                continue
            batch_count = len(selected)
            if not batch_count:
                continue
            batch_mean = float(selected.mean())
            batch_m2 = float(np.square(selected - batch_mean).sum())
            current_count = int(self.count[feature_id])
            if current_count == 0:
                self.count[feature_id] = batch_count
                self.mean[feature_id] = batch_mean
                self.m2[feature_id] = batch_m2
                continue
            total = current_count + batch_count
            delta = batch_mean - self.mean[feature_id]
            self.mean[feature_id] += delta * batch_count / total
            self.m2[feature_id] += batch_m2 + delta * delta * current_count * batch_count / total
            self.count[feature_id] = total

    def finish(self):
        mean = np.full(len(self.count), np.nan, dtype=np.float64)
        std = np.full(len(self.count), np.nan, dtype=np.float64)
        observed = self.count > 0
        mean[observed] = self.mean[observed]
        std[observed] = np.sqrt(self.m2[observed] / self.count[observed])
        mean[self.invalid] = np.nan
        std[self.invalid] = np.nan
        return mean, std


def _load_raw_sample(path):
    with np.load(path, allow_pickle=False) as sample:
        return {
            "vind": sample["vind"].astype(np.int64, copy=False),
            "minute": sample["minute"].astype(np.float64, copy=False),
            "value": sample["value"].astype(np.float64, copy=True),
            "mask": sample["mask"].astype(np.float64, copy=False),
            "x_len": int(sample["x_len"]),
            "y_len": int(sample["y_len"]),
            "ts_ind": int(sample["ts_ind"]),
            "sub_id": int(sample["sub_id"]),
        }


def _save_pickle_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _append_index(path, rows):
    if not rows:
        return
    pd.DataFrame(rows, columns=INDEX_COLUMNS).to_csv(
        path, mode="a", header=not path.exists(), index=False,
    )


def build_legacy_step4(data_dir, seed=2026):
    """Split sample metadata, fit online statistics on train context, and shard outputs."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    data_dir = Path(data_dir)
    first_root = data_dir / "first"
    index_path = first_root / "samples_index.csv"
    info = pd.read_csv(index_path)
    if info.empty:
        raise ValueError("No forecast samples found in first/samples_index.csv")
    with (data_dir / "var.pkl").open("rb") as file:
        variables, target_var = pickle.load(file)
    target_var = np.asarray(target_var, dtype=int)
    train_sub, valid_sub, test_sub = split_subjects(info["sub_id"].to_numpy(), seed=seed)
    val_model_sub, calibration_sub = split_validation_subjects(valid_sub, seed=seed)
    split_subjects_by_name = {
        "train": set(train_sub.tolist()),
        "val_model": set(val_model_sub.tolist()),
        "calibration": set(calibration_sub.tolist()),
        "test": set(test_sub.tolist()),
    }
    subject_to_split = {
        subject: split for split, subjects in split_subjects_by_name.items()
        for subject in subjects
    }
    info["split"] = info["sub_id"].map(subject_to_split)
    if info["split"].isna().any():
        raise AssertionError("A sample patient was not assigned to a split")
    info = info.sort_values(["ts_ind"], kind="mergesort").reset_index(drop=True)

    stats = OnlineFeatureStats(len(variables))
    for row in info.loc[info["split"].eq("train")].itertuples(index=False):
        sample = _load_raw_sample(first_root / row.sample_file)
        stats.update(sample["vind"][:sample["x_len"]], sample["value"][:sample["x_len"]])
    means, stds = stats.finish()

    reference_path = data_dir / "evaluation_reference_scale.pkl"
    if reference_path.is_file():
        with reference_path.open("rb") as file:
            reference = pickle.load(file)
        if reference.get("variable_names") != list(variables) or not np.array_equal(
            reference.get("target_ids"), target_var
        ):
            raise ValueError(
                "Frozen evaluation scales do not match the current target layout. "
                "Archive the old reference explicitly before creating a new benchmark scale; "
                "it will not be overwritten silently."
            )
        frozen_scales = np.asarray(reference.get("scales"), dtype=float)
        if frozen_scales.shape != (len(variables),) or not np.isfinite(frozen_scales[target_var]).all() or np.any(frozen_scales[target_var] <= 0):
            raise ValueError("Frozen evaluation scales are missing or invalid for target variables.")
    else:
        reference_scales = np.ones(len(variables), dtype=np.float64)
        for feature_id in target_var:
            if not np.isfinite(stds[feature_id]) or stds[feature_id] <= 0:
                raise ValueError(
                    "Cannot create a positive frozen evaluation scale for target {}".format(
                        variables[int(feature_id)]
                    )
                )
            reference_scales[int(feature_id)] = stds[int(feature_id)]
        reference = {
            "version": 1, "variable_names": list(variables),
            "target_ids": target_var.copy(), "scales": reference_scales,
            "source": "raw observed context values from training patients",
            "seed": int(seed),
        }
        _save_pickle_atomic(reference_path, reference)

    build_root = data_dir / ".dataset_shards_build"
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)
    split_rows = {name: 0 for name in split_subjects_by_name}
    manifest_paths = {}
    for split in split_subjects_by_name:
        split_root = build_root / split
        split_root.mkdir()
        output_index = split_root / "index.csv"
        manifest_paths[split] = "{}/index.csv".format(split)
        pending = []
        selected = info.loc[info["split"].eq(split)]
        for row in selected.itertuples(index=False):
            sample = _load_raw_sample(first_root / row.sample_file)
            feature_ids = sample["vind"]
            values = sample["value"]
            for feature_id in np.unique(feature_ids):
                mask = feature_ids == feature_id
                if np.isnan(means[feature_id]):
                    values[mask] = 0.0
                else:
                    scale = stds[feature_id] if stds[feature_id] != 0 else 1.0
                    values[mask] = (values[mask] - means[feature_id]) / scale
            output_file = "sample_{}.npz".format(int(row.ts_ind))
            temporary = split_root / (output_file + ".tmp")
            with temporary.open("wb") as file:
                np.savez_compressed(
                    file, vind=feature_ids, minute=sample["minute"], value=values,
                    mask=sample["mask"], x_len=np.asarray(sample["x_len"]),
                    y_len=np.asarray(sample["y_len"]), ts_ind=np.asarray(sample["ts_ind"]),
                    sub_id=np.asarray(sample["sub_id"]),
                )
            temporary.replace(split_root / output_file)
            pending.append({
                "ts_ind": int(row.ts_ind), "sub_id": int(row.sub_id),
                "x_len": int(row.x_len), "y_len": int(row.y_len),
                "sample_file": output_file,
            })
            split_rows[split] += 1
            if len(pending) >= 256:
                _append_index(output_index, pending)
                pending.clear()
        if pending:
            _append_index(output_index, pending)
        if not output_index.exists():
            pd.DataFrame(columns=INDEX_COLUMNS).to_csv(output_index, index=False)

    final_root = data_dir / "dataset_shards"
    backup_root = data_dir / ".dataset_shards_previous"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    if final_root.exists():
        final_root.replace(backup_root)
    build_root.replace(final_root)
    if backup_root.exists():
        shutil.rmtree(backup_root)

    _save_pickle_atomic(data_dir / "mean_std.pkl", [means, stds])
    _save_pickle_atomic(data_dir / "splits.pkl", {
        "seed": int(seed), "train_subjects": train_sub,
        "valid_subjects": val_model_sub, "model_validation_subjects": val_model_sub,
        "calibration_subjects": calibration_sub, "legacy_valid_subjects": valid_sub,
        "test_subjects": test_sub,
    })
    manifest = {
        "format": "mimiciv-sharded-v1", "root": "dataset_shards",
        "splits": manifest_paths, "sample_count_by_split": split_rows,
        "variable_names": list(variables), "target_ids": target_var.tolist(),
    }
    _save_pickle_atomic(data_dir / "dataset.pkl", manifest)
    with (data_dir / "dataset_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    report = {
        "seed": int(seed), "sample_count": int(len(info)),
        "split_counts": split_rows,
        "train_context_observations_by_variable": stats.count.tolist(),
        "nonfinite_train_features": np.flatnonzero(stats.invalid).astype(int).tolist(),
        "statistics": "online Chan/Welford population mean and std (ddof=0) over training-patient context only",
        "output_format": manifest["format"],
    }
    with (data_dir / "step_4_flow.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    return report
