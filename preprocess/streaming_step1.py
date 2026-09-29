"""Bounded-memory extraction for the legacy forecast input files."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .clinical_event_table import _normal_event_rows, load_dictionaries
    from .event_dataset import _append_link_fields, build_adult_cohort, link_event_chunk
    from .legacy_mappings import build_legacy_item_map
except ImportError:  # executed from preprocess/
    from clinical_event_table import _normal_event_rows, load_dictionaries
    from event_dataset import _append_link_fields, build_adult_cohort, link_event_chunk
    from legacy_mappings import build_legacy_item_map


CSV_COLUMNS = [
    "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "SOURCE_ROW_ID",
    "CHARTTIME", "STORETIME", "VALUENUM", "TABLE", "NAME",
]

SOURCE_COLUMNS = {
    "chartevents": [
        "subject_id", "hadm_id", "stay_id", "caregiver_id", "itemid", "charttime",
        "storetime", "value", "valuenum", "valueuom", "warning",
    ],
    "labevents": [
        "labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "charttime",
        "storetime", "value", "valuenum", "valueuom", "flag",
    ],
    "outputevents": [
        "subject_id", "hadm_id", "stay_id", "caregiver_id", "itemid", "charttime",
        "storetime", "value", "valueuom", "warning",
    ],
}


def _filter_candidate_rows(chunk, source, stay_ids, hadm_ids, subject_ids, item_ids):
    stay_values = chunk["stay_id"] if "stay_id" in chunk else pd.Series(np.nan, index=chunk.index)
    stay = pd.to_numeric(stay_values, errors="coerce")
    hadm = pd.to_numeric(chunk.get("hadm_id"), errors="coerce")
    subject = pd.to_numeric(chunk.get("subject_id"), errors="coerce")
    cohort_row = stay.isin(stay_ids) | hadm.isin(hadm_ids) | (
        subject.isin(subject_ids) & hadm.isna()
    )
    item = pd.to_numeric(chunk["itemid"], errors="coerce").isin(item_ids)
    has_measurement = chunk["charttime"].notna()
    if source in {"chartevents", "labevents"}:
        has_value = chunk["value"].notna() | pd.to_numeric(chunk["valuenum"], errors="coerce").notna()
    else:
        has_value = chunk["value"].notna()
    return chunk.loc[cohort_row & item & has_measurement & has_value].copy()


def _to_legacy_csv_rows(normalized, source):
    operational = normalized.loc[
        normalized["stay_id"].notna()
        & normalized["value"].notna()
        & normalized["variable"].ne("Temperature Site")
    ].copy()
    if operational.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)
    table_name = {"chartevents": "chart", "labevents": "lab", "outputevents": "output"}[source]
    rows = pd.DataFrame({
        "SUBJECT_ID": operational["subject_id"],
        "HADM_ID": operational["hadm_id"],
        "ICUSTAY_ID": operational["stay_id"],
        "ITEMID": operational["itemid"],
        "SOURCE_ROW_ID": operational["source_row_id"],
        "CHARTTIME": operational["measurement_time"],
        "STORETIME": operational["available_time"],
        "VALUENUM": pd.to_numeric(operational["value"], errors="coerce"),
        "TABLE": table_name,
        "NAME": operational["variable"],
    })
    return rows.reindex(columns=CSV_COLUMNS)


def _legacy_icu_frame(cohort):
    result = cohort.rename(columns={
        "subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID", "stay_id": "ICUSTAY_ID",
        "intime": "INTIME", "outtime": "OUTTIME", "anchor_age": "ANCHOR_AGE",
        "anchor_year": "ANCHOR_YEAR", "dod": "DOD", "gender": "GENDER", "age": "AGE",
    }).copy()
    return result.drop(columns=["transfer_boundary_start"], errors="ignore")


def build_legacy_step1(mimic_root, output_dir, chunksize=500_000):
    """Stream mapped chart/lab/output events into the input contract for step 2."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    mimic_root, output_dir = Path(mimic_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    icustays = pd.read_csv(mimic_root / "icu" / "icustays.csv")
    patients = pd.read_csv(mimic_root / "hosp" / "patients.csv")
    cohort, cohort_flow = build_adult_cohort(icustays, patients)
    if cohort.empty:
        raise ValueError("No adult ICU stays with valid IDs, age and intervals were found")

    d_items, d_labitems, _ = load_dictionaries(mimic_root)
    item_map = build_legacy_item_map(d_items)
    stay_ids = set(cohort["stay_id"].astype(int))
    hadm_ids = set(cohort["hadm_id"].astype(int))
    subject_ids = set(cohort["subject_id"].astype(int))
    cohort_lower = cohort.rename(columns={
        "SUBJECT_ID": "subject_id", "HADM_ID": "hadm_id", "ICUSTAY_ID": "stay_id",
        "INTIME": "intime", "OUTTIME": "outtime",
    }) if "SUBJECT_ID" in cohort else cohort

    output_tmp = output_dir / "mimic_iv_events.csv.tmp"
    icu_tmp = output_dir / "mimic_iv_icu.csv.tmp"
    output_tmp.unlink(missing_ok=True)
    icu_tmp.unlink(missing_ok=True)
    _legacy_icu_frame(cohort).to_csv(icu_tmp, index=False)

    offsets = Counter()
    row_counts = Counter()
    header_written = False
    for source, columns in SOURCE_COLUMNS.items():
        csv_path = mimic_root / ("hosp" if source == "labevents" else "icu") / (source + ".csv")
        dtype = {"value": "string", "valueuom": "string"}
        for chunk in pd.read_csv(
            csv_path, usecols=columns, chunksize=chunksize, low_memory=False, dtype=dtype,
        ):
            chunk["source_row_id"] = np.arange(
                offsets[source], offsets[source] + len(chunk), dtype=np.int64
            )
            offsets[source] += len(chunk)
            row_counts[source + "_rows_scanned"] += len(chunk)
            item_ids = {itemid for table, itemid in item_map if table == source}
            candidate = _filter_candidate_rows(
                chunk, source, stay_ids, hadm_ids, subject_ids, item_ids
            )
            row_counts[source + "_mapped_candidates"] += len(candidate)
            if candidate.empty:
                continue
            linked = link_event_chunk(candidate, cohort_lower, source)
            normalized = _normal_event_rows(linked, source, item_map, d_items, d_labitems)
            normalized = _append_link_fields(normalized, linked)
            legacy_rows = _to_legacy_csv_rows(normalized, source)
            if not legacy_rows.empty:
                legacy_rows.to_csv(
                    output_tmp, mode="a", header=not header_written, index=False,
                )
                header_written = True
                row_counts["rows_written"] += len(legacy_rows)
                row_counts[source + "_rows_written"] += len(legacy_rows)
            del normalized, linked, legacy_rows, candidate, chunk

    if not header_written:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(output_tmp, index=False)
    output_tmp.replace(output_dir / "mimic_iv_events.csv")
    icu_tmp.replace(output_dir / "mimic_iv_icu.csv")
    report = {
        "schema_version": 1,
        "chunk_rows": int(chunksize),
        "cohort": cohort_flow,
        "rows": {key: int(value) for key, value in row_counts.items()},
        "outputs": ["mimic_iv_events.csv", "mimic_iv_icu.csv"],
        "mapping_scope": "existing mapped legacy variables; inputevents remain excluded from operational forecast rows",
    }
    with (output_dir / "step_1_flow.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False, allow_nan=False)
    return report
