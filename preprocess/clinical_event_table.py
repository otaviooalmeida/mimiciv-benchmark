"""Small, provenance-preserving event-table helpers for MIMIC-IV preprocessing."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

MIT_LCP_COMMIT = "303d26c623dcc9c49cc0f204468d4acc2f063797"
MIT_LCP_VITALS_SQL = (
    "https://github.com/MIT-LCP/mimic-code/blob/"
    + MIT_LCP_COMMIT
    + "/mimic-iv/concepts_postgres/measurement/vitalsign.sql"
)
MIT_LCP_BG_SQL = (
    "https://github.com/MIT-LCP/mimic-code/blob/"
    + MIT_LCP_COMMIT
    + "/mimic-iv/concepts_postgres/measurement/bg.sql"
)

CORE_COLUMNS = [
    "subject_id", "hadm_id", "stay_id", "event_id", "table", "itemid",
    "variable", "value_raw", "unit_raw", "value", "unit",
    "measurement_time", "available_time", "source", "specimen_or_site",
    "quality_flag", "exclusion_reason", "conversion_rule_version",
]
EXTRA_COLUMNS = [
    "value_numeric_raw", "specimen_id", "item_label", "item_category",
    "dictionary_unitname", "source_row_id", "start_time", "end_time",
    "rate_raw", "rate_unit_raw", "amount_raw", "amount_unit_raw",
    "patientweight", "orderid", "linkorderid", "statusdescription",
    "isopenbag", "originalrate", "originalamount", "totalamount",
    "totalamountuom", "caregiver_id", "warning", "flag", "variable_candidates",
    "ordercategoryname", "secondaryordercategoryname", "ordercomponenttypedescription",
    "ordercategorydescription", "continueinnextdept", "cancelreason",
    "originalrateuom", "originalamountuom", "legacy_event_retained",
]
EVENT_COLUMNS = CORE_COLUMNS + EXTRA_COLUMNS


def load_dictionaries(mimic_data_dir):
    root = Path(mimic_data_dir)
    d_items_path = root / "icu" / "d_items.csv"
    d_labitems_path = root / "hosp" / "d_labitems.csv"
    d_items = pd.read_csv(
        d_items_path,
        usecols=["itemid", "label", "abbreviation", "category", "unitname", "param_type"],
    )
    d_labitems = pd.read_csv(
        d_labitems_path, usecols=["itemid", "label", "fluid", "category"]
    )
    hashes = {
        "d_items_sha256": _sha256(d_items_path),
        "d_labitems_sha256": _sha256(d_labitems_path),
    }
    return d_items, d_labitems, hashes


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variable_map_from_legacy(events):
    """Use existing item selection/name mappings without trusting its value edits."""
    table_names = {
        "chart": "chartevents", "lab": "labevents", "output": "outputevents",
        "input_mv": "inputevents", "input_weight": "inputevents",
    }
    result = {}
    mapped = events.loc[:, ["TABLE", "ITEMID", "NAME"]].dropna().drop_duplicates()
    for table, itemid, name in mapped.itertuples(index=False, name=None):
        key = (table_names.get(str(table).lower(), str(table).lower()), int(itemid))
        result.setdefault(key, set()).add(str(name))
    return result


def _source_frame(frame):
    columns = {column: str(column).lower() for column in frame.columns}
    aliases = {"icustay_id": "stay_id", "valuenum": "value_numeric_raw"}
    columns.update({column: aliases[name] for column, name in columns.items() if name in aliases})
    return frame.rename(columns=columns)


def _is_missing(value):
    return value is None or pd.isna(value)


def _unit_token(value):
    if _is_missing(value):
        return ""
    return " ".join(str(value).strip().lower().replace("°", "deg").replace(".", "").split()).replace(" ", "")


def _unit_family(value):
    token = _unit_token(value)
    return {
        "percent": "%", "percentage": "%", "f": "degf", "fahrenheit": "degf",
        "degreesf": "degf", "c": "degc", "celsius": "degc", "degreesc": "degc",
        "pounds": "lb", "pound": "lb",
        "lbs": "lb", "kilograms": "kg", "kilogram": "kg", "kgs": "kg",
        "milligrams": "mg", "micrograms": "mcg", "milliliters": "ml",
        "liters": "l", "mmhg": "mmhg",
    }.get(token, token)


def _map_variable(table, itemid, names, item_label):
    if table == "labevents" and int(itemid) == 50817:
        return "SO2_bloodgas"
    if table == "chartevents" and int(itemid) == 220277:
        return "SpO2_peripheral"
    if table == "chartevents" and int(itemid) == 224642:
        return "Temperature Site"
    if len(names) > 1:
        return "ambiguous_item_mapping"
    if names:
        name = next(iter(names))
        if name == "O2 Saturation":
            label = ("" if _is_missing(item_label) else str(item_label)).lower().replace(" ", "")
            return "SpO2_peripheral" if "pulseox" in label or "spo2" in label else "O2_saturation_chart_unclassified"
        return name
    return str(item_label) if not _is_missing(item_label) else "unmapped_item_{}".format(itemid)


def _event_source(table, itemid, variable, item_label):
    label = ("" if _is_missing(item_label) else str(item_label)).lower().replace("-", " ")
    if variable == "SO2_bloodgas":
        return "bloodgas"
    if variable == "SpO2_peripheral":
        return "peripheral_oximetry"
    if variable in {"SBP", "DBP", "MBP"}:
        if int(itemid) in {220050, 220051, 220052, 225309, 225310, 225312} or "arterial" in label or " art " in " " + label + " ":
            return "arterial_invasive"
        if int(itemid) in {220179, 220180, 220181} or any(token in label for token in ("non invasive", "noninvasive", "nibp", "nbp", "non-invasive")):
            return "noninvasive"
        return "blood_pressure_source_unclassified"
    if variable == "Temperature Site":
        return "temperature_site"
    return table


def _flags(*flags):
    return ";".join(dict.fromkeys(flag for flag in flags if flag)) or None


def _normal_event_rows(raw, table, variable_map, d_items, d_labitems):
    data = _source_frame(raw)
    if "itemid" not in data or data.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    data["itemid"] = pd.to_numeric(data["itemid"], errors="coerce")
    data = data.loc[data["itemid"].notna()].copy()
    data["itemid"] = data["itemid"].astype(int)
    dictionary = d_labitems if table == "labevents" else d_items
    data = data.merge(dictionary, on="itemid", how="left", suffixes=("", "_dictionary"), validate="many_to_one")
    names_by_item = {itemid: names for (source, itemid), names in variable_map.items() if source == table}
    variable_by_item = {}
    source_by_item = {}
    labels_by_item = data.drop_duplicates("itemid").set_index("itemid")["label"].to_dict() if "label" in data else {}
    for itemid in data["itemid"].unique():
        label = labels_by_item.get(itemid)
        variable = _map_variable(table, itemid, names_by_item.get(itemid, set()), label)
        variable_by_item[itemid] = variable
        source_by_item[itemid] = _event_source(table, itemid, variable, label)
    data["variable"] = data["itemid"].map(variable_by_item)
    data["source"] = data["itemid"].map(source_by_item)

    numeric = pd.to_numeric(data.get("value_numeric_raw"), errors="coerce")
    raw_unit = data.get("valueuom", pd.Series(None, index=data.index, dtype=object))
    unit_token = raw_unit.map(_unit_token)
    dictionary_unit = data.get("unitname", pd.Series(None, index=data.index, dtype=object))
    dictionary_mismatch = (
        raw_unit.notna() & dictionary_unit.notna()
        & raw_unit.map(_unit_family).ne(dictionary_unit.map(_unit_family))
    )
    normalized = numeric.astype(object)
    unit = raw_unit.astype(object)
    rule = pd.Series("identity-v1", index=data.index, dtype=object)
    issue = pd.Series(None, index=data.index, dtype=object)
    conversion_specs = {
        "Temperature": (
            {"f", "degf", "fahrenheit", "degreesf"},
            {"c", "degc", "celsius", "degreesc"},
            lambda values: (values - 32.0) * 5.0 / 9.0,
            "°C", "temperature-f-to-c-v1", "temperature-c-identity-v1",
            "temperature-unit-unverified-v1", "temperature_unit_unverified",
        ),
        "Weight": (
            {"lb", "lbs", "pound", "pounds"},
            {"kg", "kgs", "kilogram", "kilograms"},
            lambda values: values * 0.45359237,
            "kg", "weight-lb-to-kg-v1", "weight-kg-identity-v1",
            "weight-unit-unverified-v1", "weight_unit_unverified",
        ),
        "Height": (
            {"in", "inch", "inches"},
            {"cm", "centimeter", "centimeters"},
            lambda values: values * 2.54,
            "cm", "height-in-to-cm-v1", "height-cm-identity-v1",
            "height-unit-unverified-v1", "height_unit_unverified",
        ),
        "FiO2": (
            {"%", "percent", "percentage"},
            {"fraction", "ratio", "0-1", "1"},
            lambda values: values / 100.0,
            "fraction", "fio2-percent-to-fraction-v1", "fio2-fraction-identity-v1",
            "fio2-unit-unverified-v1", "fio2_unit_unverified",
        ),
    }
    for variable, (source_units, identity_units, convert, canonical, convert_rule, identity_rule, unknown_rule, reason) in conversion_specs.items():
        selected = data["variable"].eq(variable) & numeric.notna()
        convert_mask = selected & unit_token.isin(source_units) & ~dictionary_mismatch
        identity_mask = selected & unit_token.isin(identity_units) & ~dictionary_mismatch
        unknown_mask = selected & ~(convert_mask | identity_mask)
        normalized.loc[convert_mask] = convert(numeric.loc[convert_mask])
        normalized.loc[convert_mask | identity_mask] = normalized.loc[convert_mask | identity_mask]
        unit.loc[convert_mask | identity_mask] = canonical
        rule.loc[convert_mask] = convert_rule
        rule.loc[identity_mask] = identity_rule
        normalized.loc[unknown_mask] = np.nan
        unit.loc[unknown_mask] = None
        rule.loc[unknown_mask] = unknown_rule
        issue.loc[unknown_mask] = reason
    dictionary_conversion_mismatch = dictionary_mismatch & data["variable"].isin(conversion_specs)
    issue.loc[dictionary_conversion_mismatch] = issue.loc[dictionary_conversion_mismatch].map(
        lambda reason: "{};dictionary_unit_disagreement".format(reason)
        if not _is_missing(reason) else "dictionary_unit_disagreement"
    )

    raw_value = data.get("value", pd.Series(None, index=data.index, dtype=object)).where(
        data.get("value", pd.Series(None, index=data.index, dtype=object)).notna(),
        data.get("value_numeric_raw"),
    )
    normalized.loc[numeric.isna()] = data.get("value", pd.Series(None, index=data.index, dtype=object)).loc[numeric.isna()]
    measured = pd.to_datetime(data.get("charttime"), errors="coerce")
    available = pd.to_datetime(data.get("storetime"), errors="coerce")
    flags = pd.Series("", index=data.index, dtype=object)
    for mask, flag in (
        (available.isna(), "availability_missing"),
        (measured.isna(), "measurement_time_missing"),
        (raw_unit.isna(), "unit_missing"),
        (available.notna() & measured.notna() & (available < measured), "available_before_measurement_time"),
        (issue.notna(), "unit_unverified"),
        (data["variable"].eq("ambiguous_item_mapping"), "ambiguous_item_mapping"),
    ):
        flags.loc[mask] = flags.loc[mask].map(lambda old: _flags(old, flag) or "")
    flags.loc[dictionary_mismatch] = flags.loc[dictionary_mismatch].map(
        lambda old: _flags(old, "dictionary_unit_disagreement") or ""
    )
    quality = flags.replace("", None)
    source_row_id = data.get("source_row_id", data.get("labevent_id", pd.Series(None, index=data.index)))
    event_id = data.get("labevent_id", pd.Series(None, index=data.index)).where(
        data.get("labevent_id", pd.Series(None, index=data.index)).notna(),
        table + ":" + source_row_id.astype(str),
    )
    site_or_fluid = data.get("fluid", pd.Series(None, index=data.index, dtype=object))
    site_rows = data["variable"].eq("Temperature Site")
    site_or_fluid.loc[site_rows] = raw_value.loc[site_rows]
    exclusion = issue.copy()
    available_before_measurement = available.notna() & measured.notna() & (available < measured)
    exclusion.loc[available_before_measurement] = "available_before_measurement_time"
    ambiguous_mapping = data["variable"].eq("ambiguous_item_mapping")
    exclusion.loc[ambiguous_mapping] = "ambiguous_legacy_item_mapping"
    unlinked = table == "labevents" and data.get("stay_id", pd.Series(np.nan, index=data.index)).isna()
    if isinstance(unlinked, (bool, np.bool_)):
        unlinked = data.get("stay_id", pd.Series(np.nan, index=data.index)).isna() if unlinked else pd.Series(False, index=data.index)
    if unlinked.any():
        reasons = data.get("linkage_reason", pd.Series("outside_icu_interval_or_no_match", index=data.index))
        exclusion.loc[unlinked] = reasons.loc[unlinked]
        quality.loc[unlinked] = quality.loc[unlinked].map(lambda old: _flags(old, "stay_unlinked"))

    result = pd.DataFrame(index=data.index)
    result["subject_id"] = data.get("subject_id")
    result["hadm_id"] = data.get("hadm_id")
    result["stay_id"] = data.get("stay_id")
    result["event_id"] = event_id
    result["table"] = table
    result["itemid"] = data["itemid"]
    result["variable"] = data["variable"]
    result["value_raw"] = raw_value
    result["unit_raw"] = raw_unit
    result["value"] = normalized
    result["unit"] = unit
    result["measurement_time"] = measured
    result["available_time"] = available
    result["source"] = data["source"]
    result["specimen_or_site"] = site_or_fluid
    result["quality_flag"] = quality
    result["exclusion_reason"] = exclusion
    result["conversion_rule_version"] = rule
    result["value_numeric_raw"] = data.get("value_numeric_raw")
    result["specimen_id"] = data.get("specimen_id")
    result["item_label"] = data.get("label")
    result["item_category"] = data.get("category")
    result["dictionary_unitname"] = data.get("unitname")
    result["source_row_id"] = source_row_id
    result["caregiver_id"] = data.get("caregiver_id")
    result["warning"] = data.get("warning")
    result["flag"] = data.get("flag")
    result["variable_candidates"] = data["itemid"].map(
        lambda itemid: ";".join(sorted(names_by_item.get(itemid, set())))
    )
    return result.reindex(columns=EVENT_COLUMNS).reset_index(drop=True)


def _link_labs_to_stays(labs, icu):
    selected = _source_frame(labs)
    if "source_row_id" not in selected:
        selected["source_row_id"] = selected["labevent_id"]
    stays = _source_frame(icu[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]])
    selected["charttime"] = pd.to_datetime(selected["charttime"], errors="coerce")
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
    stays = stays.rename(columns={"stay_id": "candidate_stay_id"})
    candidates = selected.merge(stays[["hadm_id", "candidate_stay_id", "intime", "outtime"]], on="hadm_id", how="left")
    in_stay = candidates.loc[
        candidates["candidate_stay_id"].notna()
        & (candidates["charttime"] >= candidates["intime"])
        & (candidates["charttime"] <= candidates["outtime"])
    ]
    counts = in_stay.groupby("source_row_id")["candidate_stay_id"].nunique()
    matches = in_stay.drop_duplicates("source_row_id").set_index("source_row_id")["candidate_stay_id"]
    selected["stay_id"] = selected["source_row_id"].map(matches)
    selected["linkage_reason"] = selected["source_row_id"].map(counts).map(
        lambda count: "ambiguous_icu_stay_match" if count and count > 1 else None
    )
    selected.loc[selected["stay_id"].isna() & selected["linkage_reason"].isna(), "linkage_reason"] = "outside_icu_interval_or_no_match"
    selected.loc[selected["linkage_reason"].notna(), "stay_id"] = np.nan
    return selected


def _annotate_legacy_status(rows, legacy_events, source_table, source_id_column):
    if source_id_column in legacy_events:
        retained = legacy_events.loc[legacy_events["TABLE"].eq(source_table), source_id_column].dropna()
        retained_ids = set(retained.tolist())
    else:
        retained_ids = set()
    missing = ~rows["source_row_id"].isin(retained_ids)
    rows["legacy_event_retained"] = ~missing
    rows.loc[missing, "quality_flag"] = rows.loc[missing, "quality_flag"].map(
        lambda flags: _flags(flags, "not_in_legacy_mimic_iv_events")
    )
    no_prior_reason = missing & rows["exclusion_reason"].isna()
    rows.loc[no_prior_reason, "exclusion_reason"] = "excluded_from_legacy_mimic_iv_events"
    return rows


def build_chart_lab_events(chart, labs, legacy_events, icu, d_items, d_labitems):
    variable_map = variable_map_from_legacy(legacy_events)
    chart_ids = {itemid for table, itemid in variable_map if table == "chartevents"}
    chart_ids.add(224642)  # MIT-LCP temperature-site item; preserve as a separate event.
    lab_ids = {itemid for table, itemid in variable_map if table == "labevents"}
    chart_selected = chart.loc[chart["ITEMID"].isin(chart_ids)].copy()
    lab_selected = labs.loc[labs["ITEMID"].isin(lab_ids)].copy()
    lab_selected = _link_labs_to_stays(lab_selected, icu)
    chart_rows = _normal_event_rows(chart_selected, "chartevents", variable_map, d_items, d_labitems)
    lab_rows = _normal_event_rows(lab_selected, "labevents", variable_map, d_items, d_labitems)
    chart_rows = _annotate_legacy_status(chart_rows, legacy_events, "chart", "SOURCE_ROW_ID")
    lab_rows = _annotate_legacy_status(lab_rows, legacy_events, "lab", "LABEVENT_ID")
    site_rows = chart_rows.loc[chart_rows["variable"] == "Temperature Site"]
    if len(site_rows):
        site_values = site_rows.dropna(subset=["stay_id", "measurement_time"]).groupby(
            ["stay_id", "measurement_time"]
        )["value_raw"].agg(lambda values: tuple(pd.unique(values.dropna().astype(str))))
        sites = site_values.loc[site_values.map(len) == 1].map(lambda values: values[0])
        ambiguous_sites = site_values.loc[site_values.map(len) > 1]
        keys = pd.MultiIndex.from_frame(chart_rows[["stay_id", "measurement_time"]])
        site_by_event = sites.reindex(keys).to_numpy()
        temperature = chart_rows["variable"].eq("Temperature")
        chart_rows.loc[temperature, "specimen_or_site"] = site_by_event[temperature.to_numpy()]
        ambiguous_site_match = pd.Series(keys.isin(ambiguous_sites.index), index=chart_rows.index) & temperature
        chart_rows.loc[ambiguous_site_match, "quality_flag"] = chart_rows.loc[
            ambiguous_site_match, "quality_flag"
        ].map(lambda flags: _flags(flags, "temperature_site_ambiguous"))
    return pd.concat([chart_rows, lab_rows], ignore_index=True).reindex(columns=EVENT_COLUMNS)


def build_output_events(outputevents, legacy_events, d_items, d_labitems):
    variables = variable_map_from_legacy(legacy_events)
    selected = _source_frame(outputevents)
    item_ids = {itemid for table, itemid in variables if table == "outputevents"}
    selected = selected.loc[selected["itemid"].isin(item_ids)].copy()
    rows = _normal_event_rows(selected, "outputevents", variables, d_items, d_labitems)
    return _annotate_legacy_status(rows, legacy_events, "output", "SOURCE_ROW_ID")


def build_input_event_rows(inputevents, legacy_events, d_items, d_labitems):
    variables = variable_map_from_legacy(legacy_events)
    data = _source_frame(inputevents)
    item_ids = {itemid for table, itemid in variables if table == "inputevents"}
    data = data.loc[data["itemid"].isin(item_ids)].copy()
    if data.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    data = data.merge(d_items, on="itemid", how="left", validate="many_to_one")
    names_by_item = {itemid: names for (table, itemid), names in variables.items() if table == "inputevents"}
    variable_by_item = {}
    for itemid in data["itemid"].unique():
        names = names_by_item.get(itemid, set())
        variable_by_item[itemid] = "ambiguous_item_mapping" if len(names) > 1 else next(iter(names))
    data["variable"] = data["itemid"].map(variable_by_item)
    data["variable_candidates"] = data["itemid"].map(
        lambda itemid: ";".join(sorted(names_by_item.get(itemid, set())))
    )
    use_rate = data["rate"].notna()
    raw_value = data["rate"].where(use_rate, data["amount"])
    raw_unit = data["rateuom"].where(use_rate, data["amountuom"])
    unit_disagreement = (
        raw_unit.notna() & data["unitname"].notna()
        & raw_unit.map(_unit_family).ne(data["unitname"].map(_unit_family))
    )
    ambiguous = data["variable"].eq("ambiguous_item_mapping")
    flags = pd.Series("", index=data.index, dtype=object)
    flag_masks = (
        (data["storetime"].isna(), "availability_missing"),
        (data["starttime"].isna(), "measurement_time_missing"),
        (raw_value.isna(), "value_missing"),
        (raw_unit.isna(), "unit_missing"),
        (unit_disagreement, "dictionary_unit_disagreement"),
        (ambiguous, "ambiguous_item_mapping"),
        (pd.Series(True, index=data.index), "infusion_state_not_reconstructed"),
    )
    for mask, flag in flag_masks:
        flags.loc[mask] = flags.loc[mask].map(lambda old: _flags(old, flag) or "")
    result = pd.DataFrame(index=data.index)
    for column in ("subject_id", "hadm_id", "stay_id", "caregiver_id"):
        result[column] = data.get(column)
    result["event_id"] = "inputevents:" + data["source_row_id"].astype(str)
    result["table"] = "inputevents"
    result["itemid"] = data["itemid"].astype(int)
    result["variable"] = data["variable"]
    result["value_raw"] = raw_value
    result["unit_raw"] = raw_unit
    result["value"] = raw_value
    result["unit"] = raw_unit
    result["measurement_time"] = data["starttime"]
    result["available_time"] = data["storetime"]
    result["source"] = np.where(use_rate, "inputevents_rate", "inputevents_amount")
    result["quality_flag"] = flags.replace("", None)
    result["exclusion_reason"] = "inputevent_type_and_asof_state_not_audited"
    result["legacy_event_retained"] = False
    result.loc[ambiguous, "exclusion_reason"] += ";ambiguous_legacy_item_mapping"
    result["conversion_rule_version"] = "identity-raw-inputevents-v1"
    result["value_numeric_raw"] = raw_value
    for source, target in (
        ("item_label", "label"), ("item_category", "category"),
        ("dictionary_unitname", "unitname"), ("source_row_id", "source_row_id"),
        ("start_time", "starttime"), ("end_time", "endtime"),
        ("rate_raw", "rate"), ("rate_unit_raw", "rateuom"),
        ("amount_raw", "amount"), ("amount_unit_raw", "amountuom"),
        ("patientweight", "patientweight"), ("orderid", "orderid"),
        ("linkorderid", "linkorderid"), ("statusdescription", "statusdescription"),
        ("isopenbag", "isopenbag"), ("originalrate", "originalrate"),
        ("originalamount", "originalamount"), ("totalamount", "totalamount"),
        ("totalamountuom", "totalamountuom"),
        ("originalrateuom", "originalrateuom"),
        ("originalamountuom", "originalamountuom"),
        ("cancelreason", "cancelreason"),
        ("ordercategoryname", "ordercategoryname"),
        ("secondaryordercategoryname", "secondaryordercategoryname"),
        ("ordercomponenttypedescription", "ordercomponenttypedescription"),
        ("ordercategorydescription", "ordercategorydescription"),
        ("continueinnextdept", "continueinnextdept"),
    ):
        result[source] = data.get(target)
    return result.reindex(columns=EVENT_COLUMNS).reset_index(drop=True)


def append_event_rows(path, rows, append=False):
    output_path = Path(path)
    if rows.empty:
        if not output_path.exists():
            pd.DataFrame(columns=EVENT_COLUMNS).to_csv(output_path, index=False)
        return
    rows.reindex(columns=EVENT_COLUMNS).to_csv(
        output_path, index=False, mode="a" if append else "w",
        header=(not append or not output_path.exists()),
    )


def summarize_event_rows(rows):
    if rows.empty:
        return {"row_count": 0, "missing_available_time": 0, "by_variable": {}}
    grouped = rows.groupby(["table", "variable"], dropna=False).size()
    summary = {
        "row_count": int(len(rows)),
        "missing_available_time": int(rows["available_time"].isna().sum()),
        "legacy_event_retained": int(rows["legacy_event_retained"].fillna(False).sum()),
        "not_in_legacy_event_dataset": int((~rows["legacy_event_retained"].fillna(False)).sum()),
        "exclusion_reasons": {str(reason): int(count) for reason, count in rows["exclusion_reason"].value_counts(dropna=False).items()},
        "by_variable": {"{}|{}".format(table, variable): int(count) for (table, variable), count in grouped.items()},
    }
    if {"rate_unit_raw", "amount_unit_raw"}.issubset(rows.columns):
        units = rows.groupby(["variable", "rate_unit_raw", "amount_unit_raw"], dropna=False).size()
        summary["raw_rate_amount_unit_counts"] = {
            "{}|rate={}|amount={}".format(variable, rate_unit, amount_unit): int(count)
            for (variable, rate_unit, amount_unit), count in units.items()
        }
    return summary


def merge_summaries(destination, source):
    for key, value in source.items():
        if isinstance(value, dict):
            nested = destination.setdefault(key, {})
            for nested_key, count in value.items():
                nested[nested_key] = nested.get(nested_key, 0) + count
        else:
            destination[key] = destination.get(key, 0) + value
    return destination


def filter_available_at(events, cutoff, missing_policy="exclude"):
    """Return events known by cutoff and counts; missing availability is excluded by default."""
    if missing_policy not in {"exclude", "measurement_time"}:
        raise ValueError("missing_policy must be 'exclude' or explicit 'measurement_time'")
    data = events.copy()
    measurement = pd.to_datetime(data["measurement_time"], errors="coerce")
    available = pd.to_datetime(data["available_time"], errors="coerce")
    cutoff = pd.Timestamp(cutoff)
    measured_by_cutoff = measurement.notna() & (measurement <= cutoff)
    known_available = available.notna()
    available_by_cutoff = known_available & (available <= cutoff)
    fallback = (~known_available) & measured_by_cutoff & (missing_policy == "measurement_time")
    visible = measured_by_cutoff & (available_by_cutoff | fallback)
    result = data.loc[visible].copy()
    result["availability_fallback_used"] = fallback.loc[visible].to_numpy()
    if result["availability_fallback_used"].any():
        result.loc[result["availability_fallback_used"], "quality_flag"] = result.loc[
            result["availability_fallback_used"], "quality_flag"
        ].map(lambda flag: _flags(flag, "availability_fallback_measurement_time"))
    audit = {
        "input_count": int(len(data)),
        "included_count": int(visible.sum()),
        "measurement_after_cutoff": int((~measured_by_cutoff).sum()),
        "availability_after_cutoff": int((measured_by_cutoff & known_available & ~available_by_cutoff).sum()),
        "availability_missing": int((measured_by_cutoff & ~known_available).sum()),
        "availability_fallback_included": int(fallback.sum()),
        "missing_policy": missing_policy,
    }
    return result, audit


def write_audit_report(path, counts, dictionary_hashes):
    report = {
        "schema_version": 1,
        "mimic_data_version": os.environ.get("MIMIC_DATA_VERSION", "unspecified"),
        "dictionary_hashes": dictionary_hashes,
        "event_id_policy": "native labevent_id when available; source-file row ordinal for chart/output/input rows",
        "scope": "items mapped by the current preprocessing feature lists, plus temperature-site item 224642",
        "mit_lcp_reference_commit": MIT_LCP_COMMIT,
        "mit_lcp_reference_sql": [MIT_LCP_VITALS_SQL, MIT_LCP_BG_SQL],
        "reference_scope": "item/source guidance only; not universal clinical truth",
        "counts": counts,
        "availability_policy": "measurement_time <= cutoff and available_time <= cutoff; missing availability excluded by default",
        "availability_sensitivity": "filter_available_at(..., missing_policy='measurement_time') explicitly flags fallback rows",
        "infusion_policy": "raw start/rate/amount/end/status retained; no retrospective rate expansion; excluded from operational legacy features",
        "unit_policy": "raw units retained; conversions only for recognized raw units; unverified conversion yields null normalized value and exclusion reason",
    }
    with Path(path).open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False, allow_nan=False)
