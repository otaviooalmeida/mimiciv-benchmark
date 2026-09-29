"""Shared target extraction and predictive metrics for the TDSTF pipeline."""

from collections import namedtuple

import numpy as np


TargetBatch = namedtuple("TargetBatch", "values feature_ids mask")
METRIC_KINDS = {
    "MSE_mean": "squared",
    "MAE_median": "linear",
    "CRPS_empirical": "linear",
    "CRPS_fair": "linear",
    "twCRPS_upper": "linear",
    "twCRPS_lower": "linear",
    "coverage_80": "coverage",
    "width_80": "linear",
    "interval_score_80": "linear",
    "coverage_95": "coverage",
    "width_95": "linear",
    "interval_score_95": "linear",
}
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


def crps_ensemble(draws, observation, variant="empirical"):
    """Compute empirical or fair CRPS over the final (ensemble) axis in O(S log S)."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim < 1 or draws.shape[-1] < 1:
        raise ValueError("draws must contain at least one ensemble member")
    if variant not in {"empirical", "fair"}:
        raise ValueError("variant must be 'empirical' or 'fair'")
    n_draws = draws.shape[-1]
    if variant == "fair" and n_draws < 2:
        raise ValueError("fair CRPS requires at least two ensemble members")
    observation = np.asarray(observation, dtype=float)
    if not np.isfinite(draws).all() or not np.isfinite(observation).all():
        raise ValueError("draws and observations must be finite")
    observation = np.broadcast_to(observation, draws.shape[:-1])
    ordered = np.sort(draws, axis=-1)
    ranks = 2 * np.arange(1, n_draws + 1) - n_draws - 1
    pair_sum = np.sum(ordered * ranks, axis=-1)
    denominator = n_draws ** 2 if variant == "empirical" else n_draws * (n_draws - 1)
    return np.mean(np.abs(draws - observation[..., None]), axis=-1) - pair_sum / denominator


def ensemble_pit_ranks(draws, observation):
    """Return deterministic mid-rank PIT values and discrete ensemble ranks."""
    draws = np.asarray(draws, dtype=float)
    observation = np.asarray(observation, dtype=float)
    if draws.ndim < 1 or draws.shape[-1] < 1:
        raise ValueError("draws must contain at least one ensemble member")
    observation = np.broadcast_to(observation, draws.shape[:-1])
    if not np.isfinite(draws).all() or not np.isfinite(observation).all():
        raise ValueError("draws and observations must be finite")
    less = np.sum(draws < observation[..., None], axis=-1)
    equal = np.sum(draws == observation[..., None], axis=-1)
    pit = (less + 0.5 * equal) / draws.shape[-1]
    ranks = np.floor(pit * (draws.shape[-1] + 1)).astype(int)
    return pit, np.clip(ranks, 0, draws.shape[-1])


def twcrps_upper(draws, observation, threshold, variant="empirical"):
    """Upper-tail CRPS: CRPS(max(X,u), max(y,u)); score all supplied cases."""
    return crps_ensemble(np.maximum(draws, threshold), np.maximum(observation, threshold), variant)


def twcrps_lower(draws, observation, threshold, variant="empirical"):
    """Lower-tail CRPS: CRPS(min(X,l), min(y,l)); score all supplied cases."""
    return crps_ensemble(np.minimum(draws, threshold), np.minimum(observation, threshold), variant)


def _stat(numerator, denominator, reason=None, count=None):
    numerator = None if numerator is None else float(numerator)
    denominator = float(denominator)
    if count is None and denominator.is_integer():
        count = int(denominator)
    result = {
        "numerator": numerator,
        "denominator": denominator,
        "count": count,
        "value": numerator / denominator if denominator and numerator is not None else None,
    }
    if result["value"] is None:
        result["reason"] = reason or "zero_denominator"
    return result


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
    if not len(actual):
        return {
            "numerator": None, "denominator": 0.0, "count": 0,
            "value": None, "reason": "no_valid_observations",
        }
    numerator = 0.0
    for quantile in quantiles:
        prediction = np.quantile(draws, quantile, axis=-1)
        pinball = np.abs((prediction - actual) * ((actual <= prediction).astype(float) - quantile))
        numerator += 2.0 * float(pinball.sum())
    numerator /= len(quantiles)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "count": int(len(actual)),
        "value": numerator / denominator if denominator else None,
        **({} if denominator else {"reason": "zero_absolute_target_sum"}),
    }


def calculate_predictive_metrics(
    generation,
    samples_y,
    variable_names,
    target_ids,
    means,
    stds,
    reference_scales=None,
    info=None,
    tail_thresholds=None,
    forecast_origin_minute=30.0,
    include_horizons=True,
    include_patient_statistics=False,
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
    raw_samples_y = _numpy(samples_y)
    query_minutes = raw_samples_y[:, 1, :]
    if not np.isfinite(query_minutes[valid]).all():
        raise ValueError("Non-finite query time at a valid target position")

    names = list(variable_names)
    tail_thresholds = tail_thresholds or {}
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
        signal_thresholds = tail_thresholds.get(name, {})
        result = {
            "unit": unit,
            "n_observations": count,
            "tail_thresholds": signal_thresholds,
            "metrics": {},
            "diagnostics": {
                "pit": {"count": 0, "histogram": [0] * 10, "rank_histogram": []},
                "event_reliability": {},
                "quantiles_by_query_minute": {},
            },
        }
        if count == 0:
            for metric_name, kind in METRIC_KINDS.items():
                original_unit = f"{unit}²" if kind == "squared" and unit else "proportion" if kind == "coverage" else unit
                metric = {
                    "original_unit": original_unit,
                    "original_units": _stat(None, 0, reason="no_valid_observations"),
                }
                if scales is not None:
                    metric["reference_scaled"] = _stat(None, 0, reason="no_valid_observations")
                result["metrics"][metric_name] = metric
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
        n_draws = draws.shape[-1]
        pit, ranks = ensemble_pit_ranks(draws, actual)
        pit_histogram = np.histogram(pit, bins=np.linspace(0, 1, 11))[0]
        rank_histogram = np.bincount(ranks, minlength=n_draws + 1)
        query_values = query_minutes[signal_mask]
        quantile_levels = (0.05, 0.25, 0.50, 0.75, 0.95)
        query_quantiles = {}
        for query_minute in np.unique(query_values):
            selected = query_values == query_minute
            quantiles = np.quantile(draws[selected], quantile_levels, axis=-1)
            query_quantiles[str(float(query_minute))] = {
                "count": int(selected.sum()),
                "observed_sum": float(actual[selected].sum()),
                "predicted_quantile_sums": {
                    str(level): float(quantiles[index].sum())
                    for index, level in enumerate(quantile_levels)
                },
            }
        event_reliability = {}
        for tail, threshold_name in (("upper", "upper"), ("lower", "lower")):
            if threshold_name not in signal_thresholds:
                continue
            threshold = float(signal_thresholds[threshold_name])
            probabilities = (draws >= threshold).mean(axis=-1) if tail == "upper" else (draws <= threshold).mean(axis=-1)
            observed_events = (actual >= threshold) if tail == "upper" else (actual <= threshold)
            bin_indices = np.minimum((probabilities * 10).astype(int), 9)
            bins = []
            for bin_index in range(10):
                selected = bin_indices == bin_index
                bins.append({
                    "count": int(selected.sum()),
                    "predicted_probability_sum": float(probabilities[selected].sum()),
                    "observed_event_sum": int(observed_events[selected].sum()),
                })
            event_reliability[tail] = {"threshold": threshold, "bins": bins}
        result["diagnostics"] = {
            "pit": {
                "count": count,
                "histogram": pit_histogram.tolist(),
                "rank_histogram": rank_histogram.tolist(),
            },
            "event_reliability": event_reliability,
            "quantiles_by_query_minute": query_quantiles,
        }
        per_observation = {
            "MSE_mean": (np.square(point_mean - actual), "squared"),
            "MAE_median": (np.abs(point_median - actual), "linear"),
            "CRPS_empirical": (crps_ensemble(draws, actual, "empirical"), "linear"),
        }
        undefined_metrics = {
            "twCRPS_upper": "threshold_not_configured",
            "twCRPS_lower": "threshold_not_configured",
        }
        if n_draws > 1:
            per_observation["CRPS_fair"] = (crps_ensemble(draws, actual, "fair"), "linear")
        else:
            undefined_metrics["CRPS_fair"] = "requires_at_least_two_draws"
        if "upper" in signal_thresholds:
            threshold = float(signal_thresholds["upper"])
            if not np.isfinite(threshold):
                raise ValueError(f"Upper-tail threshold for {name} must be finite")
            per_observation["twCRPS_upper"] = (twcrps_upper(draws, actual, threshold), "linear")
            undefined_metrics.pop("twCRPS_upper")
        if "lower" in signal_thresholds:
            threshold = float(signal_thresholds["lower"])
            if not np.isfinite(threshold):
                raise ValueError(f"Lower-tail threshold for {name} must be finite")
            per_observation["twCRPS_lower"] = (twcrps_lower(draws, actual, threshold), "linear")
            undefined_metrics.pop("twCRPS_lower")
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

        for metric_name, reason in undefined_metrics.items():
            kind = METRIC_KINDS[metric_name]
            original_unit = f"{unit}²" if kind == "squared" and unit else "proportion" if kind == "coverage" else unit
            metric_result = {
                "original_unit": original_unit,
                "original_units": _stat(None, 0, reason=reason),
            }
            if reference_scale is not None:
                metric_result["reference_scaled"] = _stat(None, 0, reason=reason)
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
    for metric_name in METRIC_KINDS:
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

    report = {
        "metric_units": "original clinical units by signal; cross-signal aggregates use frozen reference scales",
        "metric_formulas": {
            "CRPS_empirical": "mean|x-y| - sum_{s!=r}|x_s-x_r|/(2*S^2)",
            "CRPS_fair": "mean|x-y| - sum_{s!=r}|x_s-x_r|/(2*S*(S-1)); requires S>1",
            "twCRPS_upper": "empirical CRPS(max(X,u), max(y,u)) over every valid query",
            "twCRPS_lower": "empirical CRPS(min(X,l), min(y,l)) over every valid query",
        },
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
    if include_patient_statistics:
        report["patient_statistics"] = {
            metric_name: {
                str(patient_id): {"numerator": totals[0], "denominator": totals[1]}
                for patient_id, totals in values_by_patient.items()
            }
            for metric_name, values_by_patient in patient_totals.items()
        }
    if include_horizons:
        report["by_horizon_minutes"] = {}
        for query_minute in np.unique(query_minutes[valid]):
            selected = valid & (query_minutes == query_minute)
            horizon_targets = raw_samples_y.copy()
            horizon_targets[:, 3, :] = selected.astype(float)
            horizon_report = calculate_predictive_metrics(
                generation, horizon_targets, names, target_ids, means, stds,
                reference_scales=reference_scales, info=None,
                tail_thresholds=tail_thresholds,
                forecast_origin_minute=forecast_origin_minute,
                include_horizons=False,
                include_patient_statistics=False,
            )
            horizon_key = str(float(query_minute - forecast_origin_minute))
            report["by_horizon_minutes"][horizon_key] = horizon_report["by_signal"]
    return report


def legacy_nacrps_and_mse(is_test, generation, samples_y):
    """Legacy NACRPS plus mean-forecast MSE, both on valid standardized targets."""
    targets = extract_targets(samples_y)
    generation_array = validate_forecasts(generation, targets)
    values = _numpy(targets.values)
    valid = _numpy(targets.mask).astype(bool)
    nacrps_stats = legacy_nacrps_statistics(is_test, generation_array, samples_y)
    nacrps = nacrps_stats["value"]
    if valid.any():
        mse = float(np.square(generation_array.mean(axis=-1)[valid] - values[valid]).mean())
    else:
        mse = None
    return nacrps, mse
