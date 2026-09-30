"""Small causal baselines for irregular vital-sign forecasting."""

import json
from pathlib import Path

import numpy as np

from evaluation import TARGET_UNITS, calculate_predictive_metrics, extract_targets
from metrics_stream import PredictiveMetricsAccumulator


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def fit_training_means(data_loader, target_ids):
    """Estimate standardized target means from training labels only."""
    sums = {int(feature_id): 0.0 for feature_id in target_ids}
    counts = {int(feature_id): 0 for feature_id in target_ids}
    for batch in data_loader:
        targets = extract_targets(_numpy(batch["samples_y"]))
        values = _numpy(targets.values)
        feature_ids = _numpy(targets.feature_ids)
        valid = _numpy(targets.mask).astype(bool)
        for feature_id in sums:
            selected = valid & (feature_ids == feature_id)
            sums[feature_id] += float(values[selected].sum())
            counts[feature_id] += int(selected.sum())
    missing = [feature_id for feature_id, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Training split has no valid labels for target IDs {missing}")
    return {feature_id: sums[feature_id] / counts[feature_id] for feature_id in sums}


def predict_baseline(
    method,
    samples_x,
    samples_y,
    training_means,
    moving_window=3,
    trend_window=5,
    ridge=1.0,
):
    """Predict each valid query causally; return [B,L,1] and fallback/age counts."""
    if method not in {"training_mean", "persistence", "linear_trend", "moving_average"}:
        raise ValueError(f"Unknown baseline method: {method}")
    history = _numpy(samples_x)
    samples_y = _numpy(samples_y)
    targets = extract_targets(samples_y)
    if history.ndim != 3 or history.shape[1] != 4 or history.shape[0] != samples_y.shape[0]:
        raise ValueError("samples_x must have shape [B,4,Lx] and align with samples_y")
    target_values = _numpy(targets.values)
    target_ids = _numpy(targets.feature_ids)
    target_mask = _numpy(targets.mask).astype(bool)
    history_mask = np.isfinite(history[:, 3, :]) & (history[:, 3, :] > 0)
    if not np.isfinite(history[:, 0, :][history_mask]).all() or not np.isfinite(history[:, 1, :][history_mask]).all() or not np.isfinite(history[:, 2, :][history_mask]).all():
        raise ValueError("Non-finite history field at a valid context position")

    means = {int(key): float(value) for key, value in training_means.items()} if hasattr(training_means, "items") else {
        index: float(value) for index, value in enumerate(training_means)
    }
    prediction = np.zeros_like(target_values, dtype=float)
    ages = []
    missing_age_count = 0
    fallback_count = 0
    observation_count = int(target_mask.sum())
    safe_history_ids = np.where(history_mask, history[:, 0, :], -1).astype(np.int64)
    for batch_index, query_index in zip(*np.nonzero(target_mask)):
        feature_id = int(target_ids[batch_index, query_index])
        if feature_id not in means or not np.isfinite(means[feature_id]):
            raise ValueError(f"No finite training mean for target feature {feature_id}")
        query_time = samples_y[batch_index, 1, query_index]
        causal = (
            history_mask[batch_index]
            & (safe_history_ids[batch_index] == feature_id)
            & (history[batch_index, 1] <= query_time)
        )
        positions = np.flatnonzero(causal)
        if len(positions):
            order = np.argsort(history[batch_index, 1, positions])
            positions = positions[order]
            last_position = positions[-1]
            ages.append(float(query_time - history[batch_index, 1, last_position]))
        else:
            missing_age_count += 1

        if method == "training_mean":
            value = means[feature_id]
        elif not len(positions):
            value = means[feature_id]
            fallback_count += 1
        elif method == "persistence":
            value = history[batch_index, 2, positions[-1]]
        elif method == "moving_average":
            selected = positions[-moving_window:]
            value = float(history[batch_index, 2, selected].mean())
        else:
            selected = positions[-trend_window:]
            times = history[batch_index, 1, selected]
            values = history[batch_index, 2, selected]
            if len(selected) < 2:
                value = history[batch_index, 2, selected[-1]]
            else:
                centered_times = times - times.mean()
                centered_values = values - values.mean()
                slope = float(np.dot(centered_times, centered_values) / (np.dot(centered_times, centered_times) + ridge))
                value = float(values.mean() + slope * (query_time - times.mean()))
        prediction[batch_index, query_index] = value

    age_statistics = {
        "count": len(ages),
        "missing_count": missing_age_count,
        "mean": float(np.mean(ages)) if ages else None,
        "max": float(np.max(ages)) if ages else None,
        "reason": None if ages else "no_valid_history_observations",
    }
    return prediction[..., None], {
        "method": method,
        "n_valid_targets": observation_count,
        "fallback": "training_mean",
        "fallback_count": fallback_count,
        "time_since_last_measurement_minutes": age_statistics,
        "trend_ridge": ridge if method == "linear_trend" else None,
        "moving_window": moving_window if method == "moving_average" else None,
    }


def _risk_features(history, feature_id, query_time, training_mean, normalizer_mean, normalizer_std):
    mask = np.isfinite(history[3]) & (history[3] > 0)
    if not np.isfinite(history[:3, mask]).all():
        raise ValueError("Non-finite history field at a valid position while creating risk features")
    if not np.isfinite(query_time):
        raise ValueError("Non-finite query time while creating risk features")
    safe_ids = np.where(mask, history[0], -1).astype(np.int64)
    causal = mask & (safe_ids == feature_id) & (history[1] <= query_time)
    indices = np.flatnonzero(causal)
    if not len(indices):
        age = max(float(query_time), 0.0)
        return np.asarray([training_mean, training_mean, 0.0, age, 0.0]), True, age
    indices = indices[np.argsort(history[1, indices])]
    selected = indices[-5:]
    times = history[1, selected]
    normalizer_std = normalizer_std if normalizer_std != 0 else 1.0
    values = history[2, selected] * normalizer_std + normalizer_mean
    age = float(query_time - times[-1])
    if len(selected) > 1:
        centered_time = times - times.mean()
        slope = float(np.dot(centered_time, values - values.mean()) / (np.dot(centered_time, centered_time) + 1.0))
    else:
        slope = 0.0
    return np.asarray([values[-1], values[-3:].mean(), slope, age, np.log1p(len(indices))]), False, age


def _risk_rows(data_loader, variable_names, target_ids, means, stds, training_means, thresholds):
    rows = {}
    raw_train_means = {
        int(feature_id): float(
            means[int(feature_id)]
            + (stds[int(feature_id)] if stds[int(feature_id)] != 0 else 1.0)
            * training_means[int(feature_id)]
        )
        for feature_id in target_ids
    }
    for batch in data_loader:
        history_batch = _numpy(batch["samples_x"])
        samples_y = _numpy(batch["samples_y"])
        targets = extract_targets(samples_y)
        values = _numpy(targets.values)
        feature_ids = _numpy(targets.feature_ids)
        valid = _numpy(targets.mask).astype(bool)
        for batch_index, query_index in zip(*np.nonzero(valid)):
            feature_id = int(feature_ids[batch_index, query_index])
            name = variable_names[feature_id]
            for direction in ("upper", "lower"):
                threshold = thresholds.get(name, {}).get(direction)
                if threshold is None:
                    continue
                normalizer_std = stds[feature_id] if stds[feature_id] != 0 else 1.0
                original_target = values[batch_index, query_index] * normalizer_std + means[feature_id]
                event = int(original_target >= threshold) if direction == "upper" else int(original_target <= threshold)
                features, fallback, age = _risk_features(
                    history_batch[batch_index], feature_id,
                    samples_y[batch_index, 1, query_index],
                    raw_train_means[feature_id], means[feature_id], stds[feature_id],
                )
                key = f"{name}:{direction}"
                group = rows.setdefault(key, {"features": [], "labels": [], "fallback_count": 0, "ages": []})
                group["features"].append(features)
                group["labels"].append(event)
                group["fallback_count"] += int(fallback)
                group["ages"].append(age)
    return rows


def _fit_logistic(features, labels, ridge=1.0, max_iterations=50):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (features - center) / scale
    design = np.column_stack((np.ones(len(features)), standardized))
    prevalence = float(labels.mean())
    if np.all(labels == labels[0]):
        return {
            "center": center, "scale": scale, "coefficients": None,
            "prevalence": prevalence, "reason": "single_class_training_labels",
        }
    coefficients = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(max_iterations):
        logits = np.clip(design @ coefficients, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
        gradient = design.T @ (probabilities - labels) + penalty @ coefficients
        hessian = (design.T * weights) @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.norm(step) < 1e-8:
            break
    return {
        "center": center, "scale": scale, "coefficients": coefficients,
        "prevalence": prevalence, "reason": None,
    }


def _logistic_probabilities(model, features):
    features = (np.asarray(features, dtype=float) - model["center"]) / model["scale"]
    if model["coefficients"] is None:
        return np.full(len(features), model["prevalence"])
    logits = np.clip(np.column_stack((np.ones(len(features)), features)) @ model["coefficients"], -30, 30)
    return 1.0 / (1.0 + np.exp(-logits))


def _classification_summary(probabilities, labels):
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    labels = np.asarray(labels, dtype=int)
    bins = []
    indices = np.minimum((probabilities * 10).astype(int), 9)
    for index in range(10):
        selected = indices == index
        bins.append({
            "count": int(selected.sum()),
            "predicted_probability_sum": float(probabilities[selected].sum()),
            "observed_event_sum": int(labels[selected].sum()),
        })
    positives, negatives = int(labels.sum()), int((1 - labels).sum())
    if positives and negatives:
        order = np.argsort(probabilities, kind="mergesort")
        sorted_probabilities = probabilities[order]
        ranks = np.empty(len(labels), dtype=float)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and sorted_probabilities[stop] == sorted_probabilities[start]:
                stop += 1
            ranks[order[start:stop]] = (start + 1 + stop) / 2.0
            start = stop
        auc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    else:
        auc = None
    return {
        "count": int(len(labels)),
        "event_count": positives,
        "event_prevalence": float(labels.mean()) if len(labels) else None,
        "brier_score": float(np.mean((probabilities - labels) ** 2)) if len(labels) else None,
        "log_loss": float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))) if len(labels) else None,
        "roc_auc": auc,
        "reliability_bins": bins,
    }


def evaluate_baselines(
    train_loader,
    evaluation_loader,
    variable_names,
    target_ids,
    means,
    stds,
    reference_scales,
    split,
    output_path,
    tail_thresholds=None,
):
    """Evaluate fixed forecasting and threshold-risk baselines; write one JSON report."""
    training_means = fit_training_means(train_loader, target_ids)
    methods = ("training_mean", "persistence", "linear_trend", "moving_average")
    accumulators = {method: PredictiveMetricsAccumulator() for method in methods}
    operational = {
        method: {"fallback_count": 0, "n_valid_targets": 0, "age_sum": 0.0, "age_count": 0, "age_missing": 0, "age_max": None}
        for method in methods
    }
    batches = 0
    for batch in evaluation_loader:
        batches += 1
        history = _numpy(batch["samples_x"])
        target = _numpy(batch["samples_y"])
        info = _numpy(batch["info"])
        for method in methods:
            prediction, detail = predict_baseline(method, history, target, training_means)
            report = calculate_predictive_metrics(
                prediction, target, variable_names, target_ids, means, stds,
                reference_scales=reference_scales, info=info,
                tail_thresholds=tail_thresholds, include_patient_statistics=True,
            )
            accumulators[method].add(report)
            state = operational[method]
            state["fallback_count"] += detail["fallback_count"]
            state["n_valid_targets"] += detail["n_valid_targets"]
            age = detail["time_since_last_measurement_minutes"]
            state["age_sum"] += (age["mean"] or 0.0) * age["count"]
            state["age_count"] += age["count"]
            state["age_missing"] += age["missing_count"]
            if age["max"] is not None:
                state["age_max"] = age["max"] if state["age_max"] is None else max(state["age_max"], age["max"])

    if batches == 0:
        raise ValueError(f"Baseline evaluation split '{split}' contains no batches")
    results = {}
    for method in methods:
        age = operational[method]
        results[method] = {
            "source": "causal baseline using training-only target means and observed history",
            "ensemble_size": 1,
            "predictive_metrics": accumulators[method].finalize(),
            "fallback_count": age["fallback_count"],
            "n_valid_targets": age["n_valid_targets"],
            "fallback_policy": "training target mean when no causal history exists",
            "time_since_last_measurement_minutes": {
                "count": age["age_count"],
                "missing_count": age["age_missing"],
                "mean": age["age_sum"] / age["age_count"] if age["age_count"] else None,
                "max": age["age_max"],
                "reason": None if age["age_count"] else "no_valid_history_observations",
            },
        }

    risk = {"label_definition": "future target crosses configured clinical threshold at query time", "models": {}}
    if tail_thresholds:
        train_rows = _risk_rows(train_loader, variable_names, target_ids, means, stds, training_means, tail_thresholds)
        test_rows = _risk_rows(evaluation_loader, variable_names, target_ids, means, stds, training_means, tail_thresholds)
        for key, train in train_rows.items():
            if key not in test_rows:
                continue
            variable_name, direction = key.rsplit(":", 1)
            model = _fit_logistic(train["features"], train["labels"])
            test = test_rows[key]
            prevalence_probability = np.full(len(test["labels"]), model["prevalence"])
            classifier_probability = _logistic_probabilities(model, test["features"])
            risk["models"][key] = {
                "threshold": tail_thresholds[variable_name][direction],
                "threshold_unit": TARGET_UNITS.get(variable_name),
                "training_count": len(train["labels"]),
                "training_event_count": int(np.sum(train["labels"])),
                "training_prevalence": model["prevalence"],
                "classifier": {
                    "type": "L2-regularized logistic regression",
                    "features": ["last_value", "recent_mean", "causal_slope", "time_since_last", "log1p_history_count"],
                    "training_fallback_count": train["fallback_count"],
                    "evaluation_fallback_count": test["fallback_count"],
                    "reason": model["reason"],
                    **_classification_summary(classifier_probability, test["labels"]),
                },
                "training_prevalence_baseline": _classification_summary(prevalence_probability, test["labels"]),
            }

    document = {
        "split": split,
        "population": f"{split} patient partition",
        "source": "baseline models; training means and risk models fit on train only",
        "measurement_source": "MIMIC-IV preprocessed minute-level signals; event-level provenance is not retained by the current aggregation",
        "training_means_standardized": training_means,
        "baselines": results,
        "risk_baselines": risk,
    }
    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(document, file, indent=2, ensure_ascii=False, allow_nan=False)
    return document
