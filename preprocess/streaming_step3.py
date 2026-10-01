"""Spawn-safe per-stay window sampling; workers exchange paths, never DataFrames."""

import concurrent.futures
import json
import multiprocessing
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


STEP2_COLUMNS = ["ts_ind", "minute", "available_minute", "variable", "value", "hadm_id", "sub_id"]


def _sample_stay(stay_path, output_root, target_var):
    """Process one complete stay and persist its one eligible sample, if any."""
    frame = pd.read_parquet(stay_path, columns=STEP2_COLUMNS)
    frame = frame.sort_values(["minute", "variable", "available_minute"], kind="mergesort").reset_index(drop=True)
    variable_ids = {name: index for index, name in enumerate(_WORKER_VARIABLES)}
    frame["vind"] = frame["variable"].map(variable_ids).astype("int64")
    stay_index = int(frame["ts_ind"].iloc[0])
    subject_id = int(frame["sub_id"].iloc[0])
    sample_path = Path(output_root) / "sample_{}.npz".format(stay_index)
    max_t = pd.to_numeric(frame["minute"], errors="coerce").max()
    cutoff_counts = {"context_candidates": 0, "excluded_late_availability": 0, "sampled_stays": 0}

    t = 0
    while (t + 40) < max_t:
        future = frame.loc[
            (frame["minute"] >= t + 30) & (frame["minute"] < t + 40)
            & frame["vind"].isin(target_var)
        ]
        eligible = any(int((future["vind"] == feature_id).sum()) > 1 for feature_id in target_var)
        if eligible:
            history = frame.loc[(frame["minute"] >= t) & (frame["minute"] < t + 30)]
            cutoff_counts["context_candidates"] += int(len(history))
            late = history["available_minute"] > (t + 30)
            cutoff_counts["excluded_late_availability"] += int(late.sum())
            context = history.loc[~late]
            if len(context):
                target = future.loc[future["vind"].isin(target_var)]
                x_vind = context["vind"].to_numpy(dtype=np.int64)
                x_minute = context["minute"].to_numpy(dtype=np.float64) - t
                x_value = pd.to_numeric(context["value"], errors="coerce").to_numpy(dtype=np.float64)
                y_vind = target["vind"].to_numpy(dtype=np.int64)
                y_minute = target["minute"].to_numpy(dtype=np.float64) - t
                y_value = pd.to_numeric(target["value"], errors="coerce").to_numpy(dtype=np.float64)
                x_len, y_len = len(x_vind), len(y_vind)
                vind = np.concatenate((x_vind, y_vind))
                minute = np.concatenate((x_minute, y_minute))
                value = np.concatenate((x_value, y_value))
                mask = np.concatenate((np.zeros(x_len, dtype=np.float64), np.ones(y_len, dtype=np.float64)))
                temp_path = sample_path.with_suffix(".npz.tmp")
                with temp_path.open("wb") as file:
                    np.savez_compressed(
                        file, vind=vind, minute=minute, value=value, mask=mask,
                        x_len=np.asarray(x_len), y_len=np.asarray(y_len),
                        ts_ind=np.asarray(stay_index), sub_id=np.asarray(subject_id),
                    )
                temp_path.replace(sample_path)
                cutoff_counts["sampled_stays"] = 1
                return {
                    "ts_ind": stay_index, "sub_id": subject_id, "x_len": x_len,
                    "y_len": y_len, "sample_file": sample_path.name,
                    "cutoff_counts": cutoff_counts,
                }
        t += 10
    return {"ts_ind": stay_index, "sub_id": subject_id, "sample_file": None,
            "x_len": 0, "y_len": 0, "cutoff_counts": cutoff_counts}


_WORKER_VARIABLES = []


def _initialize_worker(var_path):
    global _WORKER_VARIABLES
    with Path(var_path).open("rb") as file:
        _WORKER_VARIABLES, _ = pickle.load(file)


def _worker_sample(args):
    stay_path, output_root, target_var = args
    return _sample_stay(stay_path, output_root, target_var)


def build_legacy_step3(data_dir, workers=1):
    """Sample one window per complete stay using bounded, path-based worker tasks."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    data_dir = Path(data_dir)
    with (data_dir / "var.pkl").open("rb") as file:
        variables, target_var = pickle.load(file)
    global _WORKER_VARIABLES
    _WORKER_VARIABLES = variables
    target_var = [int(value) for value in target_var]
    stay_paths = sorted((data_dir / "sets").glob("ts_ind=*/part.parquet"))
    build_root = data_dir / ".first_build"
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)
    manifest_tmp = build_root / "samples_index.csv"
    cutoff_counts = {"context_candidates": 0, "excluded_late_availability": 0, "sampled_stays": 0}
    output_rows = []
    args_iter = iter((str(path), str(build_root), target_var) for path in stay_paths)
    processed = 0
    if workers == 1:
        _initialize_worker(data_dir / "var.pkl")
        for task in args_iter:
            result = _worker_sample(task)
            processed += 1
            for key in cutoff_counts:
                cutoff_counts[key] += result["cutoff_counts"][key]
            if result["sample_file"]:
                output_rows.append({key: result[key] for key in ("ts_ind", "sub_id", "x_len", "y_len", "sample_file")})
                if len(output_rows) >= 256:
                    _append_manifest(manifest_tmp, output_rows)
                    output_rows.clear()
    else:
        # A bounded future queue prevents unbounded task/result retention.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(str(data_dir / "var.pkl"),),
        ) as executor:
            pending = set()
            for _ in range(min(workers * 2, len(stay_paths))):
                try:
                    pending.add(executor.submit(_worker_sample, next(args_iter)))
                except StopIteration:
                    break
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    result = future.result()
                    processed += 1
                    for key in cutoff_counts:
                        cutoff_counts[key] += result["cutoff_counts"][key]
                    if result["sample_file"]:
                        output_rows.append({key: result[key] for key in ("ts_ind", "sub_id", "x_len", "y_len", "sample_file")})
                    try:
                        pending.add(executor.submit(_worker_sample, next(args_iter)))
                    except StopIteration:
                        pass
                    if len(output_rows) >= 256:
                        _append_manifest(manifest_tmp, output_rows)
                        output_rows.clear()
    if output_rows:
        _append_manifest(manifest_tmp, output_rows)
    if not manifest_tmp.exists():
        pd.DataFrame(columns=["ts_ind", "sub_id", "x_len", "y_len", "sample_file"]).to_csv(manifest_tmp, index=False)

    final_root = data_dir / "first"
    backup_root = data_dir / ".first_previous"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    if final_root.exists():
        final_root.replace(backup_root)
    build_root.replace(final_root)
    if backup_root.exists():
        shutil.rmtree(backup_root)

    with (data_dir / "availability_audit.json").open("r", encoding="utf-8") as file:
        availability_audit = json.load(file)
    report = {
        "measurement_context": "[window_start, cutoff)",
        "availability_condition": "available_minute <= cutoff",
        "missing_availability_policy": availability_audit["missing_availability_policy"],
        "fallback_rows_included_by_measurement_time": availability_audit["fallback_rows_included"],
        "stay_files_processed": int(processed),
        **cutoff_counts,
        "workers": int(workers),
        "sample_manifest": "first/samples_index.csv",
    }
    with (data_dir / "availability_cutoff_audit.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    return report


def _append_manifest(path, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
