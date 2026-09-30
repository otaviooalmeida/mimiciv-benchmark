"""Chunked event linkage, causal filtering, and per-stay legacy aggregates."""

import json
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .event_dataset import link_event_chunk
except ImportError:  # executed from preprocess/
    from event_dataset import link_event_chunk


EVENT_COLUMNS = [
    "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "SOURCE_ROW_ID", "CHARTTIME",
    "STORETIME", "VALUENUM", "TABLE", "NAME",
]
TABLE_NAMES = {"chart": "chartevents", "lab": "labevents", "output": "outputevents"}
TARGET_NAMES = ["HR", "SBP", "DBP", "Temperature", "SpO2_peripheral"]


class StayPartitionSink:
    """Write bounded chunk fragments under one logical ICU-stay partition."""

    def __init__(self, root, bucket_count):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket_count = int(bucket_count)
        self.parts = Counter()
        self.rows = Counter()

    def write(self, frame):
        if frame.empty:
            return
        frame = frame.copy()
        frame["_bucket"] = frame["ts_ind"].astype("int64") % self.bucket_count
        for bucket, group in frame.groupby("_bucket", sort=False):
            bucket = int(bucket)
            bucket_dir = self.root / "bucket={:04d}".format(bucket)
            bucket_dir.mkdir(parents=True, exist_ok=True)
            part = self.parts[bucket]
            group.drop(columns=["_bucket"]).to_parquet(
                bucket_dir / "part-{:06d}.parquet".format(part), index=False,
            )
            self.parts[bucket] += 1
            self.rows[bucket] += int(len(group))


def _link_chunk(chunk, cohort, counters):
    linked_frames = []
    source_values = chunk["TABLE"].astype("string").str.lower()
    for legacy_table, source in TABLE_NAMES.items():
        source_rows = chunk.loc[source_values.eq(legacy_table)].copy()
        if source_rows.empty:
            continue
        linked = link_event_chunk(source_rows, cohort, source)
        counters["linked_rows"] += int(linked["stay_link_status"].eq("linked").sum())
        counters["unattributable_rows"] += int(linked["stay_link_status"].ne("linked").sum())
        reasons = linked.loc[linked["stay_link_status"].ne("linked"), "linkage_reason"].value_counts()
        for reason, count in reasons.items():
            counters["linkage_reasons"][str(reason)] += int(count)
        linked_frames.append(linked)
    return pd.concat(linked_frames, ignore_index=True) if linked_frames else pd.DataFrame()


def _read_icu(path):
    icu = pd.read_csv(path, low_memory=False)
    for column in ("INTIME", "OUTTIME"):
        icu[column] = pd.to_datetime(icu[column], errors="coerce")
    icu["ICUSTAY_ID"] = pd.to_numeric(icu["ICUSTAY_ID"], errors="coerce").astype("Int64")
    icu["HADM_ID"] = pd.to_numeric(icu["HADM_ID"], errors="coerce").astype("Int64")
    icu["SUBJECT_ID"] = pd.to_numeric(icu["SUBJECT_ID"], errors="coerce").astype("Int64")
    icu = icu.reset_index(drop=True)
    icu["ts_ind"] = np.arange(len(icu), dtype=np.int64)
    lower = icu.rename(columns={
        "ICUSTAY_ID": "stay_id", "HADM_ID": "hadm_id", "SUBJECT_ID": "subject_id",
        "INTIME": "intime", "OUTTIME": "outtime",
    })
    return icu, lower


def build_legacy_step2(data_dir, chunksize=250_000, missing_availability_policy="exclude",
                       max_stay_rows=2_000_000, staging_buckets=64,
                       max_staging_bucket_rows=2_000_000):
    """Create one sorted Parquet shard per complete stay without a global events frame."""
    if chunksize <= 0 or max_stay_rows <= 0 or staging_buckets <= 0 or max_staging_bucket_rows <= 0:
        raise ValueError("chunksize and partition limits must be positive")
    if missing_availability_policy not in {"exclude", "measurement_time"}:
        raise ValueError("missing_availability_policy must be exclude or measurement_time")
    data_dir = Path(data_dir)
    icu, cohort = _read_icu(data_dir / "mimic_iv_icu.csv")
    stage_root = data_dir / ".step2_staging"
    final_tmp = data_dir / ".sets_build"
    for path in (stage_root, final_tmp):
        if path.exists():
            shutil.rmtree(path)
    sink = StayPartitionSink(stage_root, staging_buckets)
    counters = defaultdict(int)
    counters["linkage_reasons"] = Counter()
    source_missing = Counter()
    variable_names = set()
    event_path = data_dir / "mimic_iv_events.csv"

    dtype = {"TABLE": "string", "NAME": "string", "VALUENUM": "string"}
    for chunk in pd.read_csv(
        event_path, usecols=EVENT_COLUMNS, chunksize=chunksize, low_memory=False, dtype=dtype,
    ):
        counters["input_rows"] += int(len(chunk))
        chunk["CHARTTIME"] = pd.to_datetime(chunk["CHARTTIME"], errors="coerce")
        chunk["STORETIME"] = pd.to_datetime(chunk["STORETIME"], errors="coerce")
        linked = _link_chunk(chunk, cohort, counters)
        if linked.empty:
            continue
        linked = linked.loc[linked["stay_link_status"].eq("linked")].copy()
        if linked.empty:
            del chunk, linked
            continue
        linked["stay_id"] = pd.to_numeric(linked["stay_id"], errors="coerce").astype("Int64")
        linked = linked.merge(
            cohort[["stay_id", "hadm_id", "subject_id", "intime", "ts_ind"]],
            on="stay_id", how="left", suffixes=("", "_cohort"), validate="many_to_one",
        )
        # The cohort is authoritative for demographic identifiers after temporal linkage.
        linked["hadm_id"] = linked["hadm_id_cohort"].combine_first(linked["hadm_id"])
        linked["subject_id"] = linked["subject_id_cohort"].combine_first(linked["subject_id"])
        linked["rel_charttime"] = (linked["charttime"] - linked["intime"]).dt.total_seconds() / 60.0
        linked["rel_storetime"] = (linked["storetime"] - linked["intime"]).dt.total_seconds() / 60.0
        missing_storetime = linked["storetime"].isna()
        missing_charttime = linked["charttime"].isna()
        before_measurement = (
            linked["storetime"].notna() & linked["charttime"].notna()
            & (linked["storetime"] < linked["charttime"])
        )
        counters["missing_storetime_rows"] += int(missing_storetime.sum())
        counters["missing_charttime_rows"] += int(missing_charttime.sum())
        counters["storetime_before_charttime_rows"] += int(before_measurement.sum())
        for (table, name), count in linked.loc[missing_storetime].groupby(
            ["table", "name"], dropna=False
        ).size().items():
            source_missing[(str(table), str(name))] += int(count)
        if missing_availability_policy == "measurement_time":
            use_fallback = missing_storetime & ~missing_charttime
            linked.loc[use_fallback, "rel_storetime"] = linked.loc[use_fallback, "rel_charttime"]
            counters["fallback_rows_included"] += int(use_fallback.sum())
        else:
            counters["excluded_missing_availability_rows"] += int(
                (missing_storetime & ~missing_charttime).sum()
            )
        counters["excluded_storetime_before_measurement_rows"] += int(before_measurement.sum())
        valid = (
            linked["rel_charttime"].notna() & linked["rel_storetime"].notna()
            & ~before_measurement
        )
        usable = linked.loc[valid].copy()
        counters["excluded_missing_time_rows"] += int(len(linked) - len(usable))
        if usable.empty:
            del chunk, linked, usable
            continue
        usable["minute"] = np.floor(usable["rel_charttime"]).astype("int64")
        usable["available_minute"] = usable["rel_storetime"].astype("float64")
        usable["value"] = pd.to_numeric(usable["valuenum"], errors="coerce")
        usable["ts_ind"] = pd.to_numeric(usable["ts_ind"], errors="coerce").astype("int64")
        variable_names.update(usable["name"].dropna().astype(str).unique())
        staged = usable.rename(columns={
            "name": "variable", "hadm_id": "hadm_id", "subject_id": "sub_id",
        })[["ts_ind", "minute", "available_minute", "variable", "value", "hadm_id", "sub_id"]]
        sink.write(staged)
        counters["staged_rows"] += int(len(staged))
        del chunk, linked, usable, staged

    if not variable_names:
        raise ValueError("No linked events with valid measurement and availability times")
    variable_names.discard("Age")
    variable_names.discard("Gender")
    variables = sorted(variable_names)
    variable_to_index = {name: index for index, name in enumerate(variables)}
    missing_targets = [name for name in TARGET_NAMES if name not in variable_to_index]
    if missing_targets:
        raise ValueError("Target variables missing from extracted events: {}".format(missing_targets))
    target_var = np.asarray([variable_to_index[name] for name in TARGET_NAMES], dtype=int)

    final_tmp.mkdir(parents=True, exist_ok=True)
    final_rows = 0
    deduplicated_rows = 0
    maximum_bucket_rows_observed = 0
    maximum_stay_rows_observed = 0
    for bucket_dir in sorted(stage_root.glob("bucket=*")):
        bucket_index = int(bucket_dir.name.split("=", 1)[1])
        rows_in_bucket = int(sink.rows[bucket_index])
        maximum_bucket_rows_observed = max(maximum_bucket_rows_observed, rows_in_bucket)
        if rows_in_bucket > max_staging_bucket_rows:
            raise MemoryError(
                "Staging bucket {} has {} rows, above --max-staging-bucket-rows={}; "
                "increase --staging-buckets or implement external aggregation before raising the limit".format(
                    bucket_index, rows_in_bucket, max_staging_bucket_rows,
                )
            )
        frames = [pd.read_parquet(path) for path in sorted(bucket_dir.glob("*.parquet"))]
        bucket_events = pd.concat(frames, ignore_index=True)
        del frames
        for stay_value, stay_events in bucket_events.groupby("ts_ind", sort=False):
            stay_index = int(stay_value)
            rows_in_stay = int(len(stay_events))
            maximum_stay_rows_observed = max(maximum_stay_rows_observed, rows_in_stay)
            if rows_in_stay > max_stay_rows:
                raise MemoryError(
                    "ICU stay {} has {} staged rows, above --max-stay-rows={}; "
                    "implement external aggregation before raising the limit".format(
                        stay_index, rows_in_stay, max_stay_rows,
                    )
                )
            stay_events = stay_events.drop_duplicates(
                subset=["ts_ind", "minute", "available_minute", "variable", "value", "hadm_id", "sub_id"]
            )
            deduplicated_rows += int(len(stay_events))
            stay_events = stay_events.groupby(
                ["ts_ind", "minute", "variable", "available_minute"], dropna=False, sort=False
            ).agg(value=("value", "mean"), hadm_id=("hadm_id", "first"), sub_id=("sub_id", "first")).reset_index()
            stay_events = stay_events.sort_values(
                ["minute", "variable", "available_minute"], kind="mergesort"
            ).reset_index(drop=True)
            final_dir = final_tmp / "ts_ind={}".format(stay_index)
            final_dir.mkdir(parents=True, exist_ok=True)
            stay_events.to_parquet(final_dir / "part.parquet", index=False)
            final_rows += int(len(stay_events))
        del bucket_events

    if final_rows == 0:
        raise ValueError("No event aggregates were written")
    for path in (data_dir / "sets",):
        if path.exists():
            shutil.rmtree(path)
    final_tmp.replace(data_dir / "sets")
    shutil.rmtree(stage_root)
    with (data_dir / "var.pkl.tmp").open("wb") as file:
        pickle.dump([variables, target_var], file)
    (data_dir / "var.pkl.tmp").replace(data_dir / "var.pkl")

    exact_duplicate_rows = int(counters["staged_rows"] - deduplicated_rows)
    aggregation_rows_reduced = int(deduplicated_rows - final_rows)
    row_flow_closes = counters["staged_rows"] == exact_duplicate_rows + aggregation_rows_reduced + final_rows
    availability_report = {
        "input_rows_after_step1_selection": int(counters["input_rows"]),
        "linked_rows": int(counters["linked_rows"]),
        "unattributable_rows": int(counters["unattributable_rows"]),
        "linkage_reasons": dict(counters["linkage_reasons"]),
        "missing_storetime_rows": int(counters["missing_storetime_rows"]),
        "missing_charttime_rows": int(counters["missing_charttime_rows"]),
        "storetime_before_charttime_rows": int(counters["storetime_before_charttime_rows"]),
        "excluded_missing_availability_rows": int(counters["excluded_missing_availability_rows"]),
        "excluded_storetime_before_measurement_rows": int(counters["excluded_storetime_before_measurement_rows"]),
        "missing_storetime_by_table_variable": {
            "{}|{}".format(table, variable): count
            for (table, variable), count in sorted(source_missing.items())
        },
        "missing_availability_policy": missing_availability_policy,
        "fallback_rows_included": int(counters["fallback_rows_included"]),
    }
    with (data_dir / "availability_audit.json").open("w", encoding="utf-8") as file:
        json.dump(availability_report, file, indent=2, ensure_ascii=False, allow_nan=False)
    report = {
        "chunk_rows": int(chunksize),
        "staging_buckets": int(staging_buckets),
        "max_staging_bucket_rows": int(max_staging_bucket_rows),
        "maximum_staging_bucket_rows_observed": int(maximum_bucket_rows_observed),
        "max_stay_rows": int(max_stay_rows),
        "maximum_stay_rows_observed": int(maximum_stay_rows_observed),
        "input_rows": int(counters["input_rows"]),
        "linked_rows": int(counters["linked_rows"]),
        "unattributable_rows": int(counters["unattributable_rows"]),
        "staged_rows": int(counters["staged_rows"]),
        "final_rows": int(final_rows),
        "exact_duplicate_rows_collapsed": exact_duplicate_rows,
        "aggregation_rows_reduced": aggregation_rows_reduced,
        "stay_partition_count": sum(1 for _ in (data_dir / "sets").glob("ts_ind=*")),
        "row_flow_closes": bool(row_flow_closes),
        "output": "sets/ts_ind=*/part.parquet",
    }
    with (data_dir / "step_2_flow.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    return report
