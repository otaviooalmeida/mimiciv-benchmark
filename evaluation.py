"""Shared target extraction and predictive metrics for the TDSTF pipeline."""

from collections import namedtuple

import numpy as np


TargetBatch = namedtuple("TargetBatch", "values feature_ids mask")
TARGET_UNITS = {
    "HR": "bpm",
    "SBP": "mmHg",
    "DBP": "mmHg",
    "Temperature": "°C",
    "O2 Saturation": "%",
}


def _is_torch(value):
    return hasattr(value, "detach") and hasattr(value, "device")


def extract_targets(samples_y):
    """Extract [B,L] values, feature IDs and valid mask from [B,4,L] targets.

    A non-finite/zero mask denotes padding or a missing target. Non-finite values
    and feature IDs are rejected only at valid positions; invalid positions are
    returned as safe zeros/-1 so callers never need NaN * 0 masking.
    """
    if getattr(samples_y, "ndim", None) != 3 or samples_y.shape[1] != 4:
        raise ValueError(
            "samples_y must have shape [B,4,L] (feature, time, value, mask); "
            f"received {getattr(samples_y, 'shape', None)}"
        )

    if _is_torch(samples_y):
        import torch

        raw_ids, raw_values, raw_mask = samples_y[:, 0, :], samples_y[:, 2, :], samples_y[:, 3, :]
        valid = torch.isfinite(raw_mask) & (raw_mask > 0)
        valid_ids = raw_ids[valid]
        valid_values = raw_values[valid]
        if not torch.isfinite(valid_values).all().item():
            raise ValueError("Non-finite target value at a valid position")
        if not torch.isfinite(valid_ids).all().item():
            raise ValueError("Non-finite target feature ID at a valid position")
        if ((valid_ids < 0) | (valid_ids != torch.round(valid_ids))).any().item():
            raise ValueError("Target feature IDs at valid positions must be non-negative integers")
        values = torch.where(valid, raw_values, torch.zeros_like(raw_values))
        feature_ids = torch.where(valid, raw_ids, torch.full_like(raw_ids, -1)).to(torch.int64)
        return TargetBatch(values, feature_ids, valid)

    samples_y = np.asarray(samples_y)
    raw_ids, raw_values, raw_mask = samples_y[:, 0, :], samples_y[:, 2, :], samples_y[:, 3, :]
    valid = np.isfinite(raw_mask) & (raw_mask > 0)
    valid_ids, valid_values = raw_ids[valid], raw_values[valid]
    if not np.isfinite(valid_values).all():
        raise ValueError("Non-finite target value at a valid position")
    if not np.isfinite(valid_ids).all():
        raise ValueError("Non-finite target feature ID at a valid position")
    if np.any((valid_ids < 0) | (valid_ids != np.rint(valid_ids))):
        raise ValueError("Target feature IDs at valid positions must be non-negative integers")
    values = np.where(valid, raw_values, 0)
    feature_ids = np.where(valid, raw_ids, -1).astype(np.int64)
    return TargetBatch(values, feature_ids, valid)


def _numpy(value):
    return value.detach().cpu().numpy() if _is_torch(value) else np.asarray(value)


def validate_forecasts(generation, targets):
    """Assert the forecast/target tensor contract and valid-position finiteness."""
    generation = _numpy(generation)
    values = _numpy(targets.values)
    valid = _numpy(targets.mask).astype(bool)
    if generation.ndim != 3 or generation.shape[:2] != values.shape or generation.shape[2] < 1:
        raise ValueError(
            "generation must have shape [B,L,S] compatible with samples_y [B,4,L]; "
            f"received {generation.shape} and target shape {values.shape}"
        )
    if not np.isfinite(generation[valid]).all():
        raise ValueError("Non-finite forecast at a valid target position")
    return generation


def _stat(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def legacy_nacrps_statistics(is_test, generation, samples_y):
    """Return numerator/denominator for the unchanged legacy NACRPS formula."""
    targets = extract_targets(samples_y)
    generation = validate_forecasts(generation, targets)
    values = _numpy(targets.values)
    valid = _numpy(targets.mask).astype(bool)
    quantiles = np.arange(0.05, 1.0, 0.05) if is_test else np.arange(0.25, 1.0, 0.25)
    actual = values[valid]
    draws = generation[valid]
    denominator = float(np.abs(actual).sum())
    if not len(actual) or denominator == 0:
        return {"numerator": 0.0, "denominator": denominator, "value": None}
    numerator = 0.0
    for quantile in quantiles:
        prediction = np.quantile(draws, quantile, axis=-1)
        pinball = np.abs((prediction - actual) * ((actual <= prediction).astype(float) - quantile))
        numerator += 2.0 * float(pinball.sum())
    numerator /= len(quantiles)
    return {"numerator": numerator, "denominator": denominator, "value": numerator / denominator}


def calculate_predictive_metrics(
    generation,
    samples_y,
    variable_names,
    target_ids,
    means,
    stds,
    reference_scales=None,
    info=None,
):
    """Return original-unit per-signal metrics and reference-scaled aggregates.

    Forecasts have shape [B,L,S], targets [B,4,L]. Aggregated cross-signal
    values are dimensionless and use the frozen reference scale; clinical-unit
    metrics remain separate by signal.
    """
    targets = extract_targets(samples_y)
    generation = validate_forecasts(generation, targets)
    values = _numpy(targets.values)
    feature_ids = _numpy(targets.feature_ids)
    valid = _numpy(targets.mask).astype(bool)

    names = list(variable_names)
    target_ids = [int(index) for index in target_ids]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target_ids must not contain duplicates")
    unknown_targets = valid & ~np.isin(feature_ids, target_ids)
    if unknown_targets.any():
        raise ValueError("Valid target feature IDs do not match target_ids metadata")
    means, stds = _numpy(means), _numpy(stds)
    scales = None if reference_scales is None else _numpy(reference_scales)
    if means.ndim != 1 or stds.shape != means.shape:
        raise ValueError("means and stds must be compatible one-dimensional normalizer arrays")
    if scales is not None and scales.shape != means.shape:
        raise ValueError("reference_scales must have the same feature layout as means/stds")

    info_array = None if info is None else _numpy(info)
    if info_array is not None and (info_array.ndim != 2 or info_array.shape[0] != generation.shape[0]):
        raise ValueError("info must have one metadata row per forecast sample")
    patient_ids = info_array[:, 3] if info_array is not None and info_array.shape[1] >= 4 else None

    levels = (0.80, 0.95)
    alpha_by_level = {level: 1.0 - level for level in levels}
    per_signal = {}
    pooled = {}
    patient_totals = {}
    n_total = 0

    for feature_id in target_ids:
        if feature_id < 0 or feature_id >= len(names) or feature_id >= len(means):
            raise ValueError(f"Target feature ID {feature_id} is outside the variable metadata")
        signal_mask = valid & (feature_ids == feature_id)
        count = int(signal_mask.sum())
        name = names[feature_id]
        unit = TARGET_UNITS.get(name)
        result = {"unit": unit, "n_observations": count, "metrics": {}}
        if count == 0:
            per_signal[name] = result
            continue

        standardized_draws = generation[signal_mask]
        standardized_actual = values[signal_mask]
        normalizer_scale = float(stds[feature_id])
        if not np.isfinite(normalizer_scale):
            raise ValueError(f"Non-finite normalizer scale for {name}")
        if normalizer_scale == 0:
            normalizer_scale = 1.0
        mean = float(means[feature_id])
        if not np.isfinite(mean):
            raise ValueError(f"Non-finite normalizer mean for {name}")
        actual = standardized_actual * normalizer_scale + mean
        draws = standardized_draws * normalizer_scale + mean
        point_mean = draws.mean(axis=-1)
        point_median = np.median(draws, axis=-1)
        sorted_draws = np.sort(draws, axis=-1)
        n_draws = draws.shape[-1]
        ranks = (2 * np.arange(1, n_draws + 1) - n_draws - 1).astype(float)
        crps = np.mean(np.abs(draws - actual[:, None]), axis=-1) - (sorted_draws @ ranks) / (n_draws ** 2)

        per_observation = {
            "MSE_mean": (np.square(point_mean - actual), "squared"),
            "MAE_median": (np.abs(point_median - actual), "linear"),
            "CRPS_ensemble": (crps, "linear"),
        }
        for level in levels:
            tail = (1.0 - level) / 2.0
            lower, upper = np.quantile(draws, [tail, 1.0 - tail], axis=-1)
            covered = ((actual >= lower) & (actual <= upper)).astype(float)
            width = upper - lower
            interval_score = width.copy()
            interval_score += (2.0 / alpha_by_level[level]) * np.maximum(lower - actual, 0)
            interval_score += (2.0 / alpha_by_level[level]) * np.maximum(actual - upper, 0)
            suffix = int(round(level * 100))
            per_observation[f"coverage_{suffix}"] = (covered, "coverage")
            per_observation[f"width_{suffix}"] = (width, "linear")
            per_observation[f"interval_score_{suffix}"] = (interval_score, "linear")

        reference_scale = None
        if scales is not None:
            reference_scale = float(scales[feature_id])
            if not np.isfinite(reference_scale) or reference_scale <= 0:
                raise ValueError(f"Frozen evaluation scale for {name} must be finite and positive")
            result["reference_scale_original_units"] = reference_scale

        signal_scaled = {}
        for metric_name, (observations, kind) in per_observation.items():
            original = _stat(np.sum(observations), count)
            original_unit = (
                f"{unit}²" if kind == "squared" and unit else
                "proportion" if kind == "coverage" else unit
            )
            metric_result = {"original_unit": original_unit, "original_units": original}
            if reference_scale is not None:
                divisor = reference_scale ** 2 if kind == "squared" else reference_scale
                if kind == "coverage":
                    divisor = 1.0
                normalized = observations / divisor
                metric_result["reference_scaled"] = _stat(np.sum(normalized), count)
                signal_scaled[metric_name] = normalized
            result["metrics"][metric_name] = metric_result

        # Preserve per-feature observation numerators for transparent aggregation.
        for metric_name, (observations, _) in per_observation.items():
            result["metrics"][metric_name]["original_units"]["denominator"] = count

        if patient_ids is not None:
            sample_positions = np.broadcast_to(np.arange(generation.shape[0])[:, None], valid.shape)[signal_mask]
            ids_for_signal = patient_ids[sample_positions]
            if not np.isfinite(ids_for_signal).all():
                raise ValueError("Patient IDs must be finite to compute patient-level metrics")
            for metric_name, metric_values in signal_scaled.items():
                for patient_id in np.unique(ids_for_signal):
                    selected = ids_for_signal == patient_id
                    patient_key = patient_id.item()
                    totals = patient_totals.setdefault(metric_name, {}).setdefault(patient_key, [0.0, 0])
                    totals[0] += float(metric_values[selected].sum())
                    totals[1] += int(selected.sum())

        if reference_scale is not None:
            for metric_name, metric_values in signal_scaled.items():
                pooled.setdefault(metric_name, []).append(metric_values)
        per_signal[name] = result
        n_total += count

    aggregates = {}
    for metric_name in (
        "MSE_mean", "MAE_median", "CRPS_ensemble",
        "coverage_80", "width_80", "interval_score_80",
        "coverage_95", "width_95", "interval_score_95",
    ):
        observations = pooled.get(metric_name, [])
        micro_numerator = sum(float(array.sum()) for array in observations)
        micro_denominator = sum(len(array) for array in observations)
        macro_values = [float(array.mean()) for array in observations if len(array)]
        patient_means = [total / count for total, count in patient_totals.get(metric_name, {}).values() if count]
        aggregates[metric_name] = {
            "micro_per_observation": _stat(micro_numerator, micro_denominator),
            "macro_per_variable": _stat(sum(macro_values), len(macro_values)),
            "mean_per_patient": _stat(sum(patient_means), len(patient_means)),
        }

    return {
        "metric_units": "original clinical units by signal; cross-signal aggregates use frozen reference scales",
        "aggregation_definitions": {
            "micro_per_observation": "ratio of summed reference-scaled observation scores to valid observation count",
            "macro_per_variable": "unweighted mean of per-variable mean scores",
            "mean_per_patient": "unweighted mean of patient means; each patient mean pools that patient's valid, reference-scaled observations",
        },
        "reference_scale_frozen": scales is not None,
        "patient_aggregation_available": patient_ids is not None and scales is not None,
        "n_observations": n_total,
        "by_signal": per_signal,
        "aggregates": aggregates,
    }


def legacy_nacrps_and_mse(is_test, generation, samples_y):
    """Legacy NACRPS plus mean-forecast MSE, both on valid standardized targets."""
    targets = extract_targets(samples_y)
    generation_array = validate_forecasts(generation, targets)
    values = _numpy(targets.values)
    valid = _numpy(targets.mask).astype(bool)
    nacrps_stats = legacy_nacrps_statistics(is_test, generation_array, samples_y)
    nacrps = nacrps_stats["value"]
    if nacrps is None:
        nacrps = float("nan")
    if valid.any():
        mse = float(np.square(generation_array.mean(axis=-1)[valid] - values[valid]).mean())
    else:
        mse = float("nan")
    return nacrps, mse
