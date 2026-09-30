"""Chunked, admission-partitioned MIMIC-IV event dataset construction."""

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .csv_stream import iter_csv_chunks
    from .clinical_event_table import (
        EVENT_COLUMNS,
        MIT_LCP_COMMIT,
        _normal_event_rows,
        load_dictionaries,
        build_input_event_rows,
    )
except ImportError:  # executed as a script from preprocess/
    from csv_stream import iter_csv_chunks
    from clinical_event_table import (
        EVENT_COLUMNS,
        MIT_LCP_COMMIT,
        _normal_event_rows,
        load_dictionaries,
        build_input_event_rows,
    )

TEMPORAL_RULE = "[intime, outtime); exactly one candidate stay or unassigned"
TARGET_DEFINITION = {
    "primary_targets": ["HR", "SBP", "DBP", "Temperature", "SpO2_peripheral"],
    "additional_validated_variables": [
        "RR", "MBP", "Temperature Site", "Weight", "Height", "FiO2",
        "O2_saturation_chart_unclassified", "SO2_bloodgas", "Urine", "Stool",
        "Chest Tube", "Gastric", "EBL", "Emesis", "Jackson-Pratt", "Residual",
        "Ultrafiltrate", "Pre-admission Output", "Norepinephrine", "Vasopressin",
    ],
    "primary_statistic": "mean per variable/source/unit/minute",
    "bin": "floor((measurement_time - ICU intime) / 1 minute)",
    "source_policy": "aggregate each source and unit independently; never pool across sources",
    "auxiliary_statistics": ["last", "minimum", "maximum", "count", "discordance"],
    "tail_policy": "no winsorization or quantile clipping; review flags do not remove valid extremes",
}


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
    "inputevents": [
        "subject_id", "hadm_id", "stay_id", "caregiver_id", "starttime", "endtime",
        "storetime", "itemid", "amount", "amountuom", "rate", "rateuom", "orderid",
        "linkorderid", "ordercategoryname", "secondaryordercategoryname",
        "ordercomponenttypedescription", "ordercategorydescription", "patientweight",
        "totalamount", "totalamountuom", "isopenbag", "continueinnextdept", "cancelreason",
        "statusdescription", "originalamount", "originalamountuom", "originalrate", "originalrateuom",
    ],
}


LINK_COLUMNS = [
    "stay_link_method", "stay_link_status", "candidate_stay_count", "boundary_relation",
]


def _lower_columns(frame):
    return frame.rename(columns={column: str(column).lower() for column in frame.columns})


def build_adult_cohort(icustays, patients):
    """Build one row per adult ICU stay; preserve source bounds and audit exclusions."""
    stays = _lower_columns(icustays.copy())
    people = _lower_columns(patients.copy())
    for column in ("subject_id", "hadm_id", "stay_id"):
        stays[column] = pd.to_numeric(stays[column], errors="coerce").astype("Int64")
    people["subject_id"] = pd.to_numeric(people["subject_id"], errors="coerce").astype("Int64")
    flow = {"icustays_input": int(len(stays))}
    missing_identifiers = stays[["subject_id", "hadm_id", "stay_id"]].isna().any(axis=1)
    flow["excluded_missing_identifiers"] = int(missing_identifiers.sum())
    stays = stays.loc[~missing_identifiers].copy()
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
    missing_bounds = stays["intime"].isna() | stays["outtime"].isna()
    reversed_bounds = stays["outtime"].notna() & stays["intime"].notna() & (stays["outtime"] <= stays["intime"])
    flow["excluded_missing_bounds"] = int(missing_bounds.sum())
    flow["excluded_nonpositive_interval"] = int(reversed_bounds.sum())
    stays = stays.loc[~(missing_bounds | reversed_bounds)].copy()
    stays = stays.merge(people, on="subject_id", how="left", validate="many_to_one")
    stays["age"] = stays["anchor_age"] + stays["intime"].dt.year - stays["anchor_year"]
    age_missing = stays["age"].isna()
    pediatric = stays["age"].notna() & (stays["age"] < 18)
    flow["excluded_pediatric"] = int(pediatric.sum())
    flow["excluded_age_missing"] = int(age_missing.sum())
    flow["age_missing"] = int(age_missing.sum())
    stays = stays.loc[~(pediatric | age_missing)].copy()
    if stays["stay_id"].duplicated().any():
        raise ValueError("icustays.stay_id must be unique")
    stays = stays.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)
    previous_max_out = stays.groupby("subject_id", sort=False)["outtime"].transform(lambda values: values.cummax().shift())
    overlaps = stays["intime"] < previous_max_out
    touching = stays["intime"].eq(previous_max_out)
    stays["transfer_boundary_start"] = touching
    flow["overlapping_stays_same_patient"] = int(overlaps.sum())
    flow["transfer_boundaries_touching"] = int(touching.sum())
    flow["cohort_stays"] = int(len(stays))
    flow["cohort_patients"] = int(stays["subject_id"].nunique())
    flow["row_flow_closes"] = bool(
        flow["icustays_input"] == flow["excluded_missing_identifiers"]
        + flow["excluded_missing_bounds"] + flow["excluded_nonpositive_interval"]
        + flow["excluded_pediatric"]
        + flow["excluded_age_missing"] + flow["cohort_stays"]
    )
    return stays, flow


CHART_ITEM_MAP = {
    "HR": [211, 220045],
    "SBP": [51, 220050, 225309, 6701, 455, 220179, 3313, 3315, 442, 3317, 3323, 3321, 224167, 227243],
    "DBP": [8368, 220051, 225310, 8555, 8441, 220180, 8502, 8440, 8503, 8504, 8507, 8506, 224643, 227242],
    "MBP": [52, 220052, 225312, 224, 6702, 224322, 456, 220181, 3312, 3314, 3316, 3322, 3320, 443],
    "RR": [618, 220210, 3603, 224689, 614, 651, 224422, 615, 224690, 619, 224688, 227860, 227918],
    "Temperature": [3655, 677, 676, 223762, 223761, 678, 679, 3654],
    "Temperature Site": [224642],
    "Weight": [224639, 226512, 226846, 763, 226531],
    "Height": [1394, 226707, 226730],
    "FiO2": [3420, 223835, 3422, 189, 727, 190],
    "SpO2_peripheral": [220277],
    "O2_saturation_chart_unclassified": [834, 8498, 220227, 646],
}
OUTPUT_FIXED_ITEMS = {
    "Ultrafiltrate": [40286],
    "Chest Tube": [226593, 226590, 226591, 226595, 226592],
    "Gastric": [40059, 40052, 226576, 226575, 226573, 40051, 226630],
    "EBL": [40064, 226626, 40491, 226629],
    "Emesis": [40067, 226571, 40490, 41015, 40427],
    "Residual": [227510, 227511, 42837, 43892, 44909, 44959],
    "Pre-admission Output": [40060, 226633],
}
INPUT_ITEM_MAP = {
    "Vasopressin": [30051, 222315],
    "Norepinephrine": [30120, 221906, 30047],
}


def _output_variable(itemid, label):
    text = "" if pd.isna(label) else str(label).lower()
    for variable, ids in OUTPUT_FIXED_ITEMS.items():
        if int(itemid) in ids:
            return variable
    categories = (
        ("Urine", ("urine", "foley", "void", "nephrostomy", "condom", "drainage bag")),
        ("Stool", ("stool", "fecal", "colostomy", "ileostomy", "rectal")),
        ("Chest Tube", ("chest tube",)),
        ("Jackson-Pratt", ("jackson",)),
    )
    for variable, terms in categories:
        if any(term in text for term in terms):
            return variable
    return None


def dictionary_variable_map(d_items, d_labitems):
    """Return only item IDs covered by the existing validated benchmark mappings."""
    mapping = {}
    for variable, item_ids in CHART_ITEM_MAP.items():
        for itemid in item_ids:
            mapping.setdefault(("chartevents", itemid), set()).add(variable)
    mapping[("labevents", 50817)] = {"SO2_bloodgas"}
    for variable, item_ids in INPUT_ITEM_MAP.items():
        for itemid in item_ids:
            mapping[("inputevents", itemid)] = {variable}
    for row in d_items.itertuples(index=False):
        variable = _output_variable(row.itemid, getattr(row, "label", None))
        if variable is not None:
            mapping[("outputevents", int(row.itemid))] = {variable}
    return mapping


def _legacy_mapping_frame(variable_map):
    names = {"chartevents": "chart", "labevents": "lab", "outputevents": "output", "inputevents": "input_mv"}
    return pd.DataFrame([
        {"TABLE": names[table], "ITEMID": itemid, "NAME": next(iter(variables))}
        for (table, itemid), variables in variable_map.items()
    ])


def _time_candidates(rows, cohort, time_column, interval=False):
    """Return unique temporal matches; overlapping candidates are deliberately unassigned."""
    if rows.empty:
        return {}, {}, pd.Series(dtype=object)
    left = rows.copy()
    left["__row"] = np.arange(len(left), dtype=np.int64)
    source_time = "starttime" if interval else time_column
    left["__time"] = pd.to_datetime(
        left[source_time], format="mixed", errors="coerce"
    )
    left["__end"] = (
        pd.to_datetime(left["endtime"], format="mixed", errors="coerce")
        if interval else pd.NaT
    )
    right = cohort[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]].rename(
        columns={"subject_id": "candidate_subject_id", "hadm_id": "candidate_hadm_id", "stay_id": "candidate_stay_id"}
    )
    candidates = []
    with_hadm = left["hadm_id"].notna()
    if with_hadm.any():
        matched = left.loc[with_hadm, ["__row", "subject_id", "hadm_id", "__time", "__end"]].merge(
            right, left_on="hadm_id", right_on="candidate_hadm_id", how="left"
        )
        subject_ok = matched["candidate_subject_id"].isna() | matched["subject_id"].isna() | matched["subject_id"].eq(matched["candidate_subject_id"])
        if interval:
            time_ok = (matched["__time"] < matched["outtime"]) & (matched["__end"].isna() | (matched["__end"] > matched["intime"]))
        else:
            time_ok = (matched["__time"] >= matched["intime"]) & (matched["__time"] < matched["outtime"])
        candidates.append(matched.loc[subject_ok & time_ok])
    without_hadm = ~with_hadm & left["subject_id"].notna()
    if without_hadm.any():
        matched = left.loc[without_hadm, ["__row", "subject_id", "hadm_id", "__time", "__end"]].merge(
            right, left_on="subject_id", right_on="candidate_subject_id", how="left"
        )
        if interval:
            time_ok = (matched["__time"] < matched["outtime"]) & (matched["__end"].isna() | (matched["__end"] > matched["intime"]))
        else:
            time_ok = (matched["__time"] >= matched["intime"]) & (matched["__time"] < matched["outtime"])
        candidates.append(matched.loc[time_ok])
    valid = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    if valid.empty:
        return {}, {}, pd.Series(dtype=object)
    counts = valid.groupby("__row")["candidate_stay_id"].nunique().to_dict()
    unique = valid.loc[valid["__row"].map(counts).eq(1)].drop_duplicates("__row").set_index("__row")
    assignments = unique["candidate_stay_id"].to_dict()
    linked_hadm = unique["candidate_hadm_id"].to_dict()
    return assignments, linked_hadm, pd.Series(counts)


def link_event_chunk(raw, cohort, table):
    """Validate source stay IDs or assign a unique half-open temporal match."""
    events = _lower_columns(raw.copy()).reset_index(drop=True)
    for column in ("subject_id", "hadm_id", "stay_id"):
        if column in events:
            events[column] = pd.to_numeric(events[column], errors="coerce").astype("Int64")
    events["hadm_id_raw"] = events.get("hadm_id")
    source_stay = events.get("stay_id", pd.Series(np.nan, index=events.index))
    time_column = "starttime" if table == "inputevents" else "charttime"
    measured = pd.to_datetime(
        events.get(time_column), format="mixed", errors="coerce"
    )
    events["stay_id"] = np.nan
    events["stay_link_method"] = None
    events["stay_link_status"] = "unattributable"
    events["candidate_stay_count"] = 0
    events["boundary_relation"] = None
    events["linkage_reason"] = None

    direct_mask = source_stay.notna()
    if direct_mask.any():
        by_stay = cohort.set_index("stay_id")
        ids = pd.to_numeric(source_stay.loc[direct_mask], errors="coerce")
        known = ids.isin(by_stay.index)
        rows = events.index[direct_mask]
        known_rows = rows[known.to_numpy()]
        known_ids = ids.loc[known].astype(int).to_numpy()
        selected = by_stay.reindex(known_ids).reset_index(drop=True)
        selected["stay_id"] = known_ids
        time_values = measured.loc[known_rows].reset_index(drop=True)
        source_subject = events.loc[known_rows, "subject_id"].reset_index(drop=True)
        source_hadm = events.loc[known_rows, "hadm_id"].reset_index(drop=True)
        subject_ok = source_subject.isna() | source_subject.eq(selected["subject_id"])
        hadm_ok = source_hadm.isna() | source_hadm.eq(selected["hadm_id"])
        if table == "inputevents":
            end = pd.to_datetime(events.loc[known_rows, "endtime"], errors="coerce").reset_index(drop=True)
            time_ok = time_values.notna() & (time_values < selected["outtime"]) & (end.isna() | (end > selected["intime"]))
        else:
            time_ok = time_values.notna() & (time_values >= selected["intime"]) & (time_values < selected["outtime"])
        valid = subject_ok & hadm_ok & time_ok
        valid_rows = known_rows[valid.to_numpy()]
        events.loc[valid_rows, "stay_id"] = selected.loc[valid, "stay_id"].to_numpy()
        events.loc[valid_rows, "hadm_id"] = selected.loc[valid, "hadm_id"].to_numpy()
        events.loc[valid_rows, "stay_link_method"] = "source_stay_id"
        events.loc[valid_rows, "stay_link_status"] = "linked"
        events.loc[valid_rows, "candidate_stay_count"] = 1
        at_start = time_values.loc[valid].to_numpy() == selected.loc[valid, "intime"].to_numpy()
        boundary_values = np.where(at_start, "at_intime_inclusive", "within_stay")
        if table == "inputevents":
            valid_end = pd.to_datetime(events.loc[valid_rows, "endtime"], errors="coerce").to_numpy()
            crosses = (time_values.loc[valid].to_numpy() < selected.loc[valid, "intime"].to_numpy()) | (
                pd.notna(valid_end) & (valid_end > selected.loc[valid, "outtime"].to_numpy())
            )
            boundary_values = np.where(crosses, "interval_crosses_stay_boundary", boundary_values)
        if "transfer_boundary_start" in selected.columns:
            transfer = selected.loc[valid, "transfer_boundary_start"].to_numpy().copy()
            if table == "inputevents":
                transfer &= ~crosses
            boundary_values = np.where(transfer, "transfer_boundary_assigned_next_stay", boundary_values)
        events.loc[valid_rows, "boundary_relation"] = boundary_values
        invalid_rows = known_rows[~valid.to_numpy()]
        for index in invalid_rows:
            pos = known_rows.get_loc(index)
            if not subject_ok.iloc[pos]:
                reason = "source_stay_subject_mismatch"
            elif not hadm_ok.iloc[pos]:
                reason = "source_stay_admission_mismatch"
            elif pd.isna(measured.loc[index]):
                reason = "measurement_time_missing"
            elif measured.loc[index] == selected.loc[pos, "outtime"]:
                reason = "at_outtime_exclusive_boundary"
            else:
                reason = "source_stay_temporal_mismatch"
            events.loc[index, "linkage_reason"] = reason
        unknown_rows = rows[~known.to_numpy()]
        events.loc[unknown_rows, "linkage_reason"] = "source_stay_id_unknown"

    fallback_mask = ~direct_mask
    if table == "inputevents":
        assignments, linked_hadm, counts = _time_candidates(events.loc[fallback_mask], cohort, time_column, interval=True)
    else:
        assignments, linked_hadm, counts = _time_candidates(events.loc[fallback_mask], cohort, time_column)
    fallback_rows = events.index[fallback_mask]
    for local_row, stay_id in assignments.items():
        index = fallback_rows[int(local_row)]
        events.loc[index, "stay_id"] = stay_id
        events.loc[index, "hadm_id"] = linked_hadm[int(local_row)]
        events.loc[index, "stay_link_method"] = (
            "patient_time_unique" if pd.isna(events.loc[index, "hadm_id_raw"]) else "admission_time_unique"
        )
        events.loc[index, "stay_link_status"] = "linked"
        events.loc[index, "candidate_stay_count"] = int(counts.get(local_row, 1))
        stay = cohort.loc[cohort["stay_id"].eq(stay_id)].iloc[0]
        if measured.loc[index] == stay["intime"]:
            previous_out = cohort.loc[
                cohort["subject_id"].eq(stay["subject_id"]) & cohort["outtime"].eq(stay["intime"])
            ]
            boundary = "transfer_boundary_assigned_next_stay" if len(previous_out) else "at_intime_inclusive"
        elif table == "inputevents" and measured.loc[index] < stay["intime"]:
            boundary = "interval_started_before_stay"
        else:
            boundary = "within_stay"
        events.loc[index, "boundary_relation"] = boundary
        events.loc[index, "linkage_reason"] = None

    assigned_local = set(assignments)
    for local_row, original_index in enumerate(fallback_rows):
        if local_row in assigned_local:
            continue
        candidate_count = int(counts.get(local_row, 0))
        events.loc[original_index, "candidate_stay_count"] = candidate_count
        if pd.isna(measured.loc[original_index]):
            reason = "measurement_time_missing"
        elif candidate_count > 1:
            reason = "overlapping_stays_ambiguous"
        else:
            reason = "no_temporal_stay_match"
            subject = events.loc[original_index, "subject_id"]
            hadm = events.loc[original_index, "hadm_id_raw"]
            possible = cohort
            if pd.notna(hadm):
                same_admission = cohort.loc[cohort["hadm_id"].eq(hadm)]
                if same_admission.empty:
                    reason = "admission_not_in_cohort"
                elif pd.notna(subject) and not same_admission["subject_id"].eq(subject).any():
                    reason = "admission_subject_mismatch"
                possible = same_admission
            elif pd.notna(subject):
                possible = possible.loc[possible["subject_id"].eq(subject)]
            else:
                reason = "patient_id_missing_for_temporal_link"
            # Mark an exclusive outtime boundary when no next stay begins at this instant.
            if possible["outtime"].eq(measured.loc[original_index]).any():
                reason = "at_outtime_exclusive_boundary"
        events.loc[original_index, "linkage_reason"] = reason

    events["stay_link_status"] = events["stay_link_status"].fillna("unattributable")
    return events


def _append_link_fields(normalized, linked):
    for column in LINK_COLUMNS:
        normalized[column] = linked[column].to_numpy()
    normalized["hadm_id_raw"] = linked["hadm_id_raw"].to_numpy()
    failed = linked["stay_link_status"].ne("linked").to_numpy()
    if failed.any():
        normalized.loc[failed, "quality_flag"] = normalized.loc[failed, "quality_flag"].map(
            lambda flags: _append_flag(flags, "stay_unattributable")
        )
        prior = normalized.loc[failed, "exclusion_reason"].astype("string").fillna("")
        linkage = linked.loc[failed, "linkage_reason"].astype("string").fillna("stay_unattributable")
        normalized.loc[failed, "exclusion_reason"] = [
            _append_flag(left, right) for left, right in zip(prior, linkage)
        ]
    return normalized


def _append_flag(existing, flag):
    if pd.isna(existing) or not existing:
        return flag
    parts = existing.split(";")
    return existing if flag in parts else existing + ";" + flag


def _exact_duplicate_key(events):
    columns = [
        "table", "subject_id", "hadm_id", "stay_id", "itemid", "variable", "source",
        "measurement_time", "available_time", "value_raw", "value_numeric_raw", "unit_raw",
        "unit", "specimen_or_site", "specimen_id", "caregiver_id", "warning", "flag",
        "start_time", "end_time", "rate_raw", "rate_unit_raw", "amount_raw", "amount_unit_raw",
        "orderid", "linkorderid", "statusdescription", "isopenbag", "cancelreason",
        "continueinnextdept", "ordercategoryname", "secondaryordercategoryname",
        "ordercomponenttypedescription", "ordercategorydescription", "patientweight",
        "originalrate", "originalrateuom", "originalamount", "originalamountuom",
        "totalamount", "totalamountuom",
    ]
    return [column for column in columns if column in events.columns]


def attach_temperature_sites(events):
    """Attach a site only for a unique label at the same stay and measurement time."""
    result = events.copy()
    if result.empty:
        return result
    sites = result.loc[result["variable"].eq("Temperature Site")].dropna(
        subset=["stay_id", "measurement_time", "value_raw"]
    )
    if sites.empty:
        return result
    grouped = sites.groupby(["stay_id", "measurement_time"], dropna=False)["value_raw"].agg(
        lambda values: tuple(pd.unique(values.astype(str)))
    )
    unique_sites = grouped.loc[grouped.map(len).eq(1)].map(lambda values: values[0])
    ambiguous_sites = grouped.loc[grouped.map(len).gt(1)]
    temp_index = result.index[result["variable"].eq("Temperature")]
    keys = pd.MultiIndex.from_frame(result.loc[temp_index, ["stay_id", "measurement_time"]])
    result.loc[temp_index, "specimen_or_site"] = unique_sites.reindex(keys).to_numpy()
    ambiguous = keys.isin(ambiguous_sites.index)
    ambiguous_index = temp_index[ambiguous]
    result.loc[ambiguous_index, "quality_flag"] = result.loc[ambiguous_index, "quality_flag"].map(
        lambda flags: _append_flag(flags, "temperature_site_ambiguous")
    )
    return result


def classify_and_deduplicate(events):
    """Collapse only exact source-record duplicates; retain IDs/counts and classify repeats."""
    if events.empty:
        return events.assign(duplicate_count=pd.Series(dtype=int))
    result = events.copy()
    if "quality_flag" not in result:
        result["quality_flag"] = None
    keys = _exact_duplicate_key(result)
    grouped = result.groupby(keys, dropna=False, sort=False)
    result["duplicate_count"] = grouped["event_id"].transform("size").astype(int)
    result["duplicate_event_ids"] = grouped["event_id"].transform(
        lambda ids: ";".join(ids.astype(str))
    )
    if "source_row_id" in result:
        result["duplicate_source_row_ids"] = grouped["source_row_id"].transform(
            lambda ids: ";".join(str(int(value)) for value in ids.dropna())
        )
    duplicate = result["duplicate_count"].gt(1)
    result = result.drop_duplicates(keys, keep="first").copy()
    result.loc[duplicate.loc[result.index], "quality_flag"] = result.loc[duplicate.loc[result.index], "quality_flag"].map(
        lambda flags: _append_flag(flags, "exact_duplicate_collapsed")
    )
    result = result.sort_values(
        ["subject_id", "stay_id", "variable", "source", "measurement_time", "available_time", "event_id"],
        na_position="last",
    ).reset_index(drop=True)
    same_time_keys = ["subject_id", "hadm_id", "stay_id", "variable", "measurement_time"]
    valid_time = result["measurement_time"].notna()
    same_time = result.loc[valid_time].groupby(same_time_keys, dropna=False, sort=False)
    source_counts = same_time["source"].transform("nunique")
    simultaneous_counts = same_time["event_id"].transform("size")
    result["simultaneous_measurement_count"] = 1
    result["measurement_relation"] = "initial_measurement"
    result["source_disagreement"] = False
    result.loc[valid_time, "simultaneous_measurement_count"] = simultaneous_counts.to_numpy()
    other_source = valid_time.copy()
    other_source.loc[valid_time] = source_counts.gt(1).to_numpy()
    same_source_repeat = valid_time.copy()
    same_source_repeat.loc[valid_time] = (source_counts.eq(1) & simultaneous_counts.gt(1)).to_numpy()
    result.loc[other_source, "measurement_relation"] = "simultaneous_other_source"
    result.loc[same_source_repeat, "measurement_relation"] = "simultaneous_repeat_same_source"
    same_time_units = result.loc[valid_time].groupby(
        same_time_keys + ["unit"], dropna=False, sort=False
    )
    unit_source_counts = same_time_units["source"].transform("nunique")
    comparable = same_time_units["value"].transform("nunique")
    result.loc[valid_time, "source_disagreement"] = (
        unit_source_counts.gt(1) & comparable.gt(1)
    ).to_numpy()
    repeat_order = result.groupby(["subject_id", "stay_id", "variable", "source"], dropna=False, sort=False).cumcount().gt(0)
    initial = result["measurement_relation"].eq("initial_measurement")
    result.loc[initial & repeat_order, "measurement_relation"] = "repeated_measurement"
    return result


def aggregate_minute_source(events, cohort):
    """Aggregate per stay/variable/source/unit/minute; mean is the fixed primary estimand."""
    if events.empty:
        return pd.DataFrame(columns=[
            "subject_id", "hadm_id", "stay_id", "table", "variable", "source", "unit", "minute",
            "event_count", "source_record_count", "valid_value_count", "value_mean", "value_last", "value_min",
            "value_max", "discordant", "distinct_value_count", "measurement_time_max", "available_time_max",
            "missing_availability_count", "invalid_availability_count", "source_disagreement",
        ])
    data = events.loc[
        events["stay_id"].notna()
        & events["measurement_time"].notna()
        & events["table"].ne("inputevents")
    ].copy()
    data = data.merge(cohort[["stay_id", "intime"]], on="stay_id", how="left", validate="many_to_one")
    data["minute"] = np.floor((data["measurement_time"] - data["intime"]).dt.total_seconds() / 60).astype("int64")
    keys = ["subject_id", "hadm_id", "stay_id", "table", "variable", "source", "unit", "minute"]
    data["_numeric"] = pd.to_numeric(data["value"], errors="coerce")
    data["_invalid_availability"] = (
        data["available_time"].notna() & data["measurement_time"].notna()
        & (data["available_time"] < data["measurement_time"])
    )
    if "duplicate_count" not in data:
        data["duplicate_count"] = 1
    data = data.sort_values([*keys, "measurement_time", "available_time", "event_id"], na_position="first")
    grouped = data.groupby(keys, dropna=False, sort=False)
    result = grouped.agg(
        event_count=("event_id", "size"),
        source_record_count=("duplicate_count", "sum"),
        valid_value_count=("_numeric", "count"),
        value_mean=("_numeric", "mean"),
        value_min=("_numeric", "min"),
        value_max=("_numeric", "max"),
        distinct_value_count=("_numeric", "nunique"),
        measurement_time_max=("measurement_time", "max"),
        available_time_max=("available_time", "max"),
        missing_availability_count=("available_time", lambda values: int(values.isna().sum())),
        invalid_availability_count=("_invalid_availability", "sum"),
        value_last=("_numeric", "last"),
        source_disagreement=("source_disagreement", "max"),
    ).reset_index()
    result["discordant"] = result["distinct_value_count"].gt(1)
    # A group containing any event without storetime is not safely available as one aggregate.
    result.loc[result["missing_availability_count"].gt(0), "available_time_max"] = pd.NaT
    return result


def aggregate_available_at(aggregates, cutoff):
    """Keep aggregates with complete, non-contradictory availability by cutoff."""
    cutoff = pd.Timestamp(cutoff)
    measurement_max = pd.to_datetime(aggregates["measurement_time_max"], errors="coerce")
    available_max = pd.to_datetime(aggregates["available_time_max"], errors="coerce")
    visible = (
        measurement_max.notna() & (measurement_max <= cutoff)
        & available_max.notna() & (available_max <= cutoff)
        & aggregates["missing_availability_count"].eq(0)
        & aggregates["invalid_availability_count"].eq(0)
    )
    return aggregates.loc[visible].copy()


def cast_event_schema(rows):
    """Enforce identical physical Parquet types across source partitions."""
    result = rows.reindex(columns=EVENT_COLUMNS).copy()
    id_columns = [
        "subject_id", "hadm_id", "stay_id", "itemid", "specimen_id", "source_row_id",
        "caregiver_id", "orderid", "linkorderid",
    ]
    numeric_columns = ["value", "value_numeric_raw", "patientweight"]
    datetime_columns = ["measurement_time", "available_time", "start_time", "end_time"]
    bool_columns = ["legacy_event_retained", "source_disagreement"]
    count_columns = ["candidate_stay_count", "duplicate_count", "simultaneous_measurement_count"]
    for column in id_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
    for column in datetime_columns:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    for column in bool_columns:
        result[column] = result[column].astype("boolean")
    result["candidate_stay_count"] = pd.to_numeric(result["candidate_stay_count"], errors="coerce").fillna(0).astype("int64")
    for column in ("duplicate_count", "simultaneous_measurement_count"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(1).astype("int64")
    non_string = set(id_columns + numeric_columns + datetime_columns + bool_columns + count_columns)
    for column in result.columns:
        if column not in non_string:
            result[column] = result[column].astype("string")
    return result


def _partition_name(hadm_id, subject_id):
    if pd.notna(hadm_id):
        return "hadm_{}".format(int(hadm_id))
    if pd.notna(subject_id):
        return "unassigned_subject_{}".format(int(subject_id))
    return "unassigned_unknown"


class PartitionSink:
    """Stage chunk rows into fixed hash buckets before admission-level finalization."""

    def __init__(self, root, bucket_count):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket_count = int(bucket_count)
        self.parts = Counter()
        self.rows = Counter()

    def write(self, rows):
        if rows.empty:
            return
        staged = rows.copy()
        hadm = pd.to_numeric(staged["hadm_id"], errors="coerce")
        subject = pd.to_numeric(staged["subject_id"], errors="coerce")
        identity = hadm.fillna(subject).fillna(-1).astype("int64")
        staged["_bucket"] = np.mod(identity, self.bucket_count)
        for bucket, group in staged.groupby("_bucket", sort=False):
            bucket = int(bucket)
            folder = self.root / "bucket={:04d}".format(bucket)
            folder.mkdir(parents=True, exist_ok=True)
            part = self.parts[bucket]
            cast_event_schema(group.drop(columns=["_bucket"])).to_parquet(
                folder / "part-{:06d}.parquet".format(part), index=False,
            )
            self.parts[bucket] += 1
            self.rows[bucket] += int(len(group))


def _write_linked_chunk(raw, table, cohort, variable_map, d_items, d_labitems, legacy_mapping):
    linked = link_event_chunk(raw, cohort, table)
    if table == "inputevents":
        normalized = build_input_event_rows(linked, legacy_mapping, d_items, d_labitems)
    else:
        normalized = _normal_event_rows(linked, table, variable_map, d_items, d_labitems)
    normalized = _append_link_fields(normalized, linked)
    if "value" in normalized:
        invalid = normalized["exclusion_reason"].astype("string").str.contains(
            "physically_impossible|non_finite|unit_unverified|dictionary_unit_disagreement", na=False
        )
        normalized.loc[invalid, "value"] = np.nan
    return normalized, linked


def _record_flow(flow, source, candidate_rows, linked_rows):
    flow[source]["candidate_rows"] += int(len(candidate_rows))
    flow[source]["normalized_rows"] += int(len(linked_rows))
    flow[source]["linked_rows"] += int(linked_rows["stay_link_status"].eq("linked").sum())
    flow[source]["unattributable_rows"] += int(linked_rows["stay_link_status"].ne("linked").sum())
    counts = linked_rows["linkage_reason"].fillna("linked").value_counts()
    for reason, count in counts.items():
        flow[source]["linkage_status_counts"][str(reason)] += int(count)
    flow[source]["source_stay_id_rows"] += int(linked_rows["stay_link_method"].eq("source_stay_id").sum())
    no_hadm = linked_rows["hadm_id_raw"].isna()
    flow[source]["rows_without_hadm_id"] += int(no_hadm.sum())
    flow[source]["patient_time_no_hadm_rows"] += int(
        linked_rows["stay_link_method"].eq("patient_time_unique").sum()
    )
    flow[source]["unlinked_without_hadm_id"] += int(
        (no_hadm & linked_rows["stay_link_status"].ne("linked")).sum()
    )
    flow[source]["boundary_intime_rows"] += int(linked_rows["boundary_relation"].eq("at_intime_inclusive").sum())
    flow[source]["transfer_boundary_rows"] += int(
        linked_rows["boundary_relation"].eq("transfer_boundary_assigned_next_stay").sum()
    )
    flow[source]["outtime_exclusive_rows"] += int(
        linked_rows["linkage_reason"].eq("at_outtime_exclusive_boundary").sum()
    )


def build_event_dataset(mimic_root, output_dir, chunksize=250_000, max_partition_rows=2_000_000,
                        staging_buckets=64, max_staging_bucket_rows=2_000_000):
    """Create chunked raw-event Parquet partitions and minute/source aggregates."""
    if min(chunksize, max_partition_rows, staging_buckets, max_staging_bucket_rows) <= 0:
        raise ValueError("chunk and partition limits must be positive")
    mimic_root = Path(mimic_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_root = output_dir / ".events_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    event_root = output_dir / "events.parquet"
    aggregate_root = output_dir / "minute_aggregates.parquet"
    for path in (event_root, aggregate_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    icustays = pd.read_csv(mimic_root / "icu" / "icustays.csv")
    patients = pd.read_csv(mimic_root / "hosp" / "patients.csv")
    cohort, cohort_flow = build_adult_cohort(icustays, patients)
    cohort_path = output_dir / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    d_items, d_labitems, dictionary_hashes = load_dictionaries(mimic_root)
    variable_map = dictionary_variable_map(d_items, d_labitems)
    legacy_mapping = _legacy_mapping_frame(variable_map)
    sinks = {
        source: PartitionSink(staging_root / source, staging_buckets)
        for source in SOURCE_COLUMNS
    }
    flow = defaultdict(lambda: {
        "rows_scanned": 0, "rows_filtered_cohort": 0, "rows_excluded_unmapped": 0,
        "candidate_rows": 0, "normalized_rows": 0,
        "linked_rows": 0, "unattributable_rows": 0,
        "linkage_status_counts": Counter(), "source_stay_id_rows": 0, "patient_time_no_hadm_rows": 0,
        "rows_without_hadm_id": 0, "unlinked_without_hadm_id": 0,
        "boundary_intime_rows": 0, "transfer_boundary_rows": 0, "outtime_exclusive_rows": 0,
    })
    offsets = Counter()
    cohort_stays = set(pd.to_numeric(cohort["stay_id"], errors="coerce").dropna().astype(int))
    cohort_hadm = set(pd.to_numeric(cohort["hadm_id"], errors="coerce").dropna().astype(int))
    cohort_subjects = set(pd.to_numeric(cohort["subject_id"], errors="coerce").dropna().astype(int))

    for source, columns in SOURCE_COLUMNS.items():
        csv_path = mimic_root / ("icu" if source != "labevents" else "hosp") / (source + ".csv")
        raw_string_columns = {
            "chartevents": {"value": "string", "valueuom": "string"},
            "labevents": {"value": "string", "valueuom": "string"},
            "outputevents": {"value": "string", "valueuom": "string"},
            "inputevents": {
                "amount": "string", "amountuom": "string", "rate": "string", "rateuom": "string",
                "originalamount": "string", "originalamountuom": "string",
                "originalrate": "string", "originalrateuom": "string",
                "totalamount": "string", "totalamountuom": "string", "statusdescription": "string",
            },
        }[source]
        for chunk in iter_csv_chunks(
            csv_path, columns, chunksize=chunksize, dtype=raw_string_columns,
            optional_columns={"warning", "flag"}.intersection(columns),
        ):
            chunk["source_row_id"] = np.arange(offsets[source], offsets[source] + len(chunk), dtype=np.int64)
            offsets[source] += len(chunk)
            flow[source]["rows_scanned"] += int(len(chunk))
            if source == "labevents":
                hadm = pd.to_numeric(chunk["hadm_id"], errors="coerce")
                subject = pd.to_numeric(chunk["subject_id"], errors="coerce")
                keep = hadm.isin(cohort_hadm) | (hadm.isna() & subject.isin(cohort_subjects))
            else:
                stay = pd.to_numeric(chunk["stay_id"], errors="coerce")
                subject = pd.to_numeric(chunk["subject_id"], errors="coerce")
                hadm = pd.to_numeric(chunk["hadm_id"], errors="coerce")
                cohort_patient_row = hadm.isin(cohort_hadm) | (
                    subject.isin(cohort_subjects) & hadm.isna()
                )
                keep = stay.isin(cohort_stays) | cohort_patient_row
            cohort_candidate = chunk.loc[keep].copy()
            flow[source]["rows_filtered_cohort"] += int(len(chunk) - len(cohort_candidate))
            mapped_items = {itemid for mapped_source, itemid in variable_map if mapped_source == source}
            candidate = cohort_candidate.loc[
                pd.to_numeric(cohort_candidate["itemid"], errors="coerce").isin(mapped_items)
            ].copy()
            flow[source]["rows_excluded_unmapped"] += int(len(cohort_candidate) - len(candidate))
            if candidate.empty:
                continue
            normalized, linked = _write_linked_chunk(
                candidate, source, cohort, variable_map, d_items, d_labitems, legacy_mapping
            )
            _record_flow(flow, source, candidate, linked)
            if len(normalized) != len(candidate):
                raise AssertionError("source-to-event row count mismatch for {}".format(source))
            sinks[source].write(normalized)
            del normalized, linked, candidate, chunk

    bucket_keys = set()
    for source in SOURCE_COLUMNS:
        source_stage = staging_root / source
        bucket_keys.update(path.name for path in source_stage.iterdir() if path.is_dir())

    source_partition_totals = defaultdict(lambda: Counter())
    partition_count = 0
    maximum_partition_rows_observed = 0
    maximum_bucket_rows_observed = 0
    for bucket_key in sorted(bucket_keys):
        staged_by_source_bucket = Counter({
            source: int(sinks[source].rows[int(bucket_key.split("=", 1)[1])])
            for source in SOURCE_COLUMNS
            if sinks[source].rows[int(bucket_key.split("=", 1)[1])]
        })
        staged_bucket_rows = sum(staged_by_source_bucket.values())
        maximum_bucket_rows_observed = max(maximum_bucket_rows_observed, staged_bucket_rows)
        if staged_bucket_rows > max_staging_bucket_rows:
            raise MemoryError(
                "Staging {} has {} rows, above --max-staging-bucket-rows={}; "
                "increase --staging-buckets or implement external aggregation before raising the limit".format(
                    bucket_key, staged_bucket_rows, max_staging_bucket_rows,
                )
            )
        frames = []
        for source in SOURCE_COLUMNS:
            folder = staging_root / source / bucket_key
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.parquet")):
                frames.append(pd.read_parquet(path))
        bucket_rows = pd.concat(frames, ignore_index=True)
        del frames
        bucket_rows["measurement_time"] = pd.to_datetime(bucket_rows["measurement_time"], errors="coerce")
        bucket_rows["available_time"] = pd.to_datetime(bucket_rows["available_time"], errors="coerce")
        for (hadm_id, subject_id), partition_rows in bucket_rows.groupby(
            ["hadm_id", "subject_id"], dropna=False, sort=False
        ):
            partition_key = _partition_name(hadm_id, subject_id)
            staged_partition_rows = int(len(partition_rows))
            maximum_partition_rows_observed = max(
                maximum_partition_rows_observed, staged_partition_rows,
            )
            if staged_partition_rows > max_partition_rows:
                raise MemoryError(
                    "Partition {} has {} staged rows, above --max-partition-rows={}; "
                    "implement external partition aggregation before raising this limit".format(
                        partition_key, staged_partition_rows, max_partition_rows,
                    )
                )
            staged_by_source = Counter(
                partition_rows.groupby("table", dropna=False).size().to_dict()
            )
            rows = attach_temperature_sites(partition_rows.copy())
            rows = classify_and_deduplicate(rows)
            rows = cast_event_schema(rows)
            aggregates = aggregate_minute_source(rows, cohort[["stay_id", "intime"]])
            rows = rows.sort_values(
                ["subject_id", "hadm_id", "stay_id", "measurement_time", "available_time", "table", "itemid", "event_id"],
                na_position="last",
            )
            aggregate_path = aggregate_root / partition_key
            event_path = event_root / partition_key
            aggregate_path.mkdir(parents=True, exist_ok=True)
            event_path.mkdir(parents=True, exist_ok=True)
            rows.to_parquet(event_path / "part-00000.parquet", index=False)
            aggregates.sort_values(
                ["stay_id", "minute", "variable", "source", "unit"], na_position="last"
            ).to_parquet(aggregate_path / "part-00000.parquet", index=False)
            final_by_source = rows.groupby("table", dropna=False).size().to_dict()
            aggregate_by_source = aggregates.groupby("table", dropna=False).size().to_dict()
            for source, source_staged in staged_by_source.items():
                source_final = int(final_by_source.get(source, 0))
                source_partition_totals[source]["staged_rows"] += int(source_staged)
                source_partition_totals[source]["final_rows"] += source_final
                source_partition_totals[source]["exact_duplicates_collapsed"] += int(source_staged - source_final)
                source_partition_totals[source]["aggregate_rows"] += int(aggregate_by_source.get(source, 0))
                source_partition_totals[source]["partition_count"] += 1
            partition_count += 1
            del rows, aggregates, partition_rows
        del bucket_rows

    # Validate closure from source candidates through staging and exact deduplication.
    for source, stats in flow.items():
        source_counts = source_partition_totals[source]
        staged_source = int(source_counts["staged_rows"])
        final_source = int(source_counts["final_rows"])
        deduplicated_source = int(source_counts["exact_duplicates_collapsed"])
        if staged_source != stats["normalized_rows"]:
            raise AssertionError("{} source rows were lost before staging closure".format(source))
        if staged_source != final_source + deduplicated_source:
            raise AssertionError("{} exact-duplicate row flow does not close".format(source))
    staged_total = sum(values["staged_rows"] for values in source_partition_totals.values())
    final_total = sum(values["final_rows"] for values in source_partition_totals.values())
    deduplicated_total = sum(values["exact_duplicates_collapsed"] for values in source_partition_totals.values())
    if staged_total != final_total + deduplicated_total:
        raise AssertionError("event row flow does not close")
    for source, stats in flow.items():
        stats["source_selection_flow_closes"] = bool(
            stats["rows_scanned"] == stats["rows_filtered_cohort"]
            + stats["rows_excluded_unmapped"] + stats["candidate_rows"]
        )
        stats["normalization_flow_closes"] = bool(stats["candidate_rows"] == stats["normalized_rows"])
        if not stats["source_selection_flow_closes"] or not stats["normalization_flow_closes"]:
            raise AssertionError("{} source row flow does not close".format(source))
    shutil.rmtree(staging_root)
    report = {
        "schema_version": 3,
        "mimic_data_version": os.environ.get("MIMIC_DATA_VERSION", "unspecified"),
        "dictionary_hashes": dictionary_hashes,
        "mapping_reference": {"repository": "MIT-LCP/mimic-code", "commit": MIT_LCP_COMMIT},
        "cohort": cohort_flow,
        "temporal_rule": TEMPORAL_RULE,
        "linkage": {source: {key: (dict(value) if isinstance(value, Counter) else value) for key, value in stats.items()} for source, stats in flow.items()},
        "partition_memory_guard": {
            "staging_buckets": int(staging_buckets),
            "max_staging_bucket_rows": int(max_staging_bucket_rows),
            "maximum_staging_bucket_rows_observed": int(maximum_bucket_rows_observed),
            "max_partition_rows": int(max_partition_rows),
            "maximum_partition_rows_observed": int(maximum_partition_rows_observed),
            "guard_basis": "fixed hash bucket and admission/subject partition row counts before materialization",
        },
        "row_flow": {
            "staged_rows": int(staged_total), "final_events": int(final_total),
            "exact_duplicates_collapsed": int(deduplicated_total),
            "closure_holds": bool(staged_total == final_total + deduplicated_total),
            "partition_count": int(partition_count),
            "by_source": {source: dict(counts) for source, counts in source_partition_totals.items()},
        },
        "mapping_scope": "only existing validated benchmark item mappings; unmapped in-cohort rows are counted and not exported",
        "target_definition": TARGET_DEFINITION,
        "availability_gate": "aggregates require all contributing availability times present and not before measurement; maximum measurement and availability times must be <= cutoff",
        "infusion_policy": "raw inputevents intervals retained; no amount/endtime rate reconstruction; no operational infusion state",
        "deduplication": {
            "key": _exact_duplicate_key(pd.DataFrame(columns=EVENT_COLUMNS)),
            "rule": "collapse only identical source/table, patient/admission/stay, item/variable/source, measurement and availability times, raw/normalized values/units, specimen/site and caregiver/status/order fields; retain duplicate_count, event IDs and source row IDs",
            "repeated_measurements": "same-time same-source values and same-time cross-source values remain separate and are classified",
        },
        "cohort_output": str(cohort_path.name),
        "event_output": str(event_root.name),
        "aggregate_output": str(aggregate_root.name),
        "variable_source_summary_output": "variable_source_summary.parquet",
        "limitations": ["Events and IDs remain local; do not commit patient-level artifacts.", "Quantiles in the variable/source report use a deterministic bounded reservoir and are labeled approximate."],
    }
    summary = summarize_variable_source(event_root, aggregate_root, cohort)
    summary.to_parquet(output_dir / "variable_source_summary.parquet", index=False)
    review_sample = sample_extreme_review_events(event_root, output_dir / "extreme_review_sample.parquet")
    report["extreme_review"] = {
        "sample_path": "extreme_review_sample.parquet",
        "sample_rows": int(len(review_sample)),
        "review_required": bool(len(review_sample)),
        "sampling": "deterministic reservoir, stratified by table, variable, source and unit, maximum 100 rows per stratum",
    }
    with (output_dir / "cohort_flow.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False, allow_nan=False)
    return report


def sample_extreme_review_events(event_root, output_path, sample_per_stratum=100):
    """Create a local, ID-bearing specialist review sample; never add it to Git."""
    import pyarrow.parquet as pq

    rng = np.random.default_rng(83017)
    samples = defaultdict(list)
    seen = Counter()
    columns = [
        "subject_id", "hadm_id", "stay_id", "event_id", "table", "itemid", "variable",
        "source", "value_raw", "unit_raw", "value", "unit", "measurement_time",
        "available_time", "quality_flag", "exclusion_reason",
    ]
    for path in sorted(Path(event_root).glob("**/*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=200_000, columns=columns):
            frame = batch.to_pandas()
            extreme = frame["quality_flag"].fillna("").str.contains("extreme_value_review")
            for _, row in frame.loc[extreme].iterrows():
                key = (row["table"], row["variable"], row["source"], row["unit"])
                seen[key] += 1
                entry = row.to_dict()
                if len(samples[key]) < sample_per_stratum:
                    samples[key].append(entry)
                else:
                    slot = int(rng.integers(0, seen[key]))
                    if slot < sample_per_stratum:
                        samples[key][slot] = entry
    rows = [row for stratum in sorted(samples, key=str) for row in samples[stratum]]
    result = pd.DataFrame(rows, columns=columns)
    result.to_parquet(output_path, index=False)
    return result


def summarize_variable_source(event_root, aggregate_root, cohort, reservoir_size=2000):
    """Summarize partitions with a deterministic bounded sample for report quantiles."""
    groups = defaultdict(lambda: {
        "raw_count": 0, "source_record_count": 0, "linked_count": 0,
        "unattributable_count": 0, "valid_count": 0, "missing_value_count": 0, "excluded_count": 0,
        "raw_min": np.inf, "raw_max": -np.inf, "value_min": np.inf, "value_max": -np.inf,
        "delay_count": 0, "delay_missing_count": 0, "delay_sum": 0.0, "delay_max": -np.inf,
        "negative_delay_count": 0, "delay_sample": [], "delay_seen": 0,
        "review_extreme_count": 0, "minute_bin_count": 0, "discordant_minute_count": 0,
        "excluded_reasons": Counter(), "sample": [], "seen": 0,
    })
    import pyarrow.parquet as pq

    rng = np.random.default_rng(2026)
    event_files = sorted(Path(event_root).glob("**/*.parquet"))
    columns = [
        "table", "variable", "source", "unit", "value", "value_numeric_raw", "measurement_time",
        "available_time", "stay_id", "duplicate_count", "exclusion_reason", "quality_flag",
    ]
    for path in event_files:
        parquet_file = pq.ParquetFile(path)
        for record_batch in parquet_file.iter_batches(batch_size=200_000, columns=columns):
            batch_frame = record_batch.to_pandas()
            for key, frame in batch_frame.groupby(["table", "variable", "source", "unit"], dropna=False):
                key = tuple(None if pd.isna(value) else value for value in key)
                state = groups[key]
                state["raw_count"] += int(len(frame))
                state["source_record_count"] += int(pd.to_numeric(frame["duplicate_count"], errors="coerce").fillna(1).sum())
                state["linked_count"] += int(frame["stay_id"].notna().sum())
                state["unattributable_count"] += int(frame["stay_id"].isna().sum())
                raw = pd.to_numeric(frame["value_numeric_raw"], errors="coerce").to_numpy(dtype=float)
                values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
                raw = raw[np.isfinite(raw)]
                values = values[np.isfinite(values)]
                if len(raw):
                    state["raw_min"] = min(state["raw_min"], float(raw.min()))
                    state["raw_max"] = max(state["raw_max"], float(raw.max()))
                if len(values):
                    state["valid_count"] += int(len(values))
                    state["value_min"] = min(state["value_min"], float(values.min()))
                    state["value_max"] = max(state["value_max"], float(values.max()))
                    for value in values:
                        state["seen"] += 1
                        if len(state["sample"]) < reservoir_size:
                            state["sample"].append(float(value))
                        else:
                            position = int(rng.integers(0, state["seen"]))
                            if position < reservoir_size:
                                state["sample"][position] = float(value)
                state["missing_value_count"] += int(pd.to_numeric(frame["value"], errors="coerce").isna().sum())
                state["review_extreme_count"] += int(
                    frame["quality_flag"].fillna("").str.contains("extreme_value_review").sum()
                )
                reasons = frame["exclusion_reason"].dropna().astype(str).value_counts()
                state["excluded_count"] += int(reasons.sum())
                for reason, count in reasons.items():
                    state["excluded_reasons"][reason] += int(count)
                measurement = pd.to_datetime(frame["measurement_time"], errors="coerce")
                available = pd.to_datetime(frame["available_time"], errors="coerce")
                delay = (available - measurement).dt.total_seconds().to_numpy(dtype=float) / 60
                finite_delay = delay[np.isfinite(delay)]
                state["delay_count"] += int(len(finite_delay))
                state["delay_missing_count"] += int(len(delay) - len(finite_delay))
                if len(finite_delay):
                    state["delay_sum"] += float(finite_delay.sum())
                    state["delay_max"] = max(state["delay_max"], float(finite_delay.max()))
                    state["negative_delay_count"] += int((finite_delay < 0).sum())
                    for value in finite_delay:
                        state["delay_seen"] += 1
                        if len(state["delay_sample"]) < reservoir_size:
                            state["delay_sample"].append(float(value))
                        else:
                            position = int(rng.integers(0, state["delay_seen"]))
                            if position < reservoir_size:
                                state["delay_sample"][position] = float(value)
    aggregate_files = sorted(Path(aggregate_root).glob("**/*.parquet"))
    aggregate_columns = ["table", "variable", "source", "unit", "discordant"]
    for path in aggregate_files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=200_000, columns=aggregate_columns):
            frame = batch.to_pandas()
            for key, group in frame.groupby(["table", "variable", "source", "unit"], dropna=False):
                key = tuple(None if pd.isna(value) else value for value in key)
                state = groups[key]
                state["minute_bin_count"] += int(len(group))
                state["discordant_minute_count"] += int(group["discordant"].fillna(False).sum())
    rows = []
    total_icu_hours = float((cohort["outtime"] - cohort["intime"]).dt.total_seconds().sum() / 3600)
    for (table, variable, source, unit), state in groups.items():
        sample = np.asarray(state["sample"], dtype=float)
        quantiles = np.quantile(sample, [0.01, 0.05, 0.5, 0.95, 0.99]) if len(sample) else [np.nan] * 5
        delay_sample = np.asarray(state["delay_sample"], dtype=float)
        delay_quantiles = np.quantile(delay_sample, [0.5, 0.95]) if len(delay_sample) else [np.nan, np.nan]
        rows.append({
            "table": table, "variable": variable, "source": source, "unit": unit,
            "deduplicated_event_count": state["raw_count"],
            "source_record_count_before_deduplication": state["source_record_count"],
            "exact_duplicate_records_collapsed": state["source_record_count"] - state["raw_count"],
            "linked_event_count": state["linked_count"],
            "unattributable_event_count": state["unattributable_count"],
            "valid_value_count": state["valid_count"],
            "missing_value_count": state["missing_value_count"], "excluded_count": state["excluded_count"],
            "review_extreme_count": state["review_extreme_count"],
            "minute_bin_count": state["minute_bin_count"],
            "discordant_minute_count": state["discordant_minute_count"],
            "frequency_per_1000_icu_hours": state["linked_count"] / total_icu_hours * 1000 if total_icu_hours else np.nan,
            "raw_min": state["raw_min"] if np.isfinite(state["raw_min"]) else np.nan,
            "raw_max": state["raw_max"] if np.isfinite(state["raw_max"]) else np.nan,
            "normalized_min": state["value_min"] if np.isfinite(state["value_min"]) else np.nan,
            "normalized_max": state["value_max"] if np.isfinite(state["value_max"]) else np.nan,
            "q01_approx": quantiles[0], "q05_approx": quantiles[1], "q50_approx": quantiles[2],
            "q95_approx": quantiles[3], "q99_approx": quantiles[4],
            "quantile_method": "deterministic_reservoir_n={}".format(len(sample)),
            "mean_recording_delay_minutes": state["delay_sum"] / state["delay_count"] if state["delay_count"] else np.nan,
            "recording_delay_q50_approx": delay_quantiles[0],
            "recording_delay_q95_approx": delay_quantiles[1],
            "max_recording_delay_minutes": state["delay_max"] if np.isfinite(state["delay_max"]) else np.nan,
            "negative_recording_delay_count": state["negative_delay_count"],
            "missing_recording_delay_count": state["delay_missing_count"],
            "exclusion_reasons": ";".join("{}={}".format(k, v) for k, v in sorted(state["excluded_reasons"].items())),
        })
    return pd.DataFrame(rows)
