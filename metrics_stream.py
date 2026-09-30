"""Merge batch-sized metric reports without retaining forecast tensors."""

from evaluation import METRIC_KINDS, _stat


class PredictiveMetricsAccumulator:
    def __init__(self):
        self.by_signal = {}
        self.by_horizon = {}
        self.patient_totals = {}
        self.reference_scale_frozen = None
        self.patient_aggregation_available = None

    @staticmethod
    def _merge_signal(destination, source):
        destination.setdefault("unit", source.get("unit"))
        destination.setdefault("tail_thresholds", source.get("tail_thresholds", {}))
        if "reference_scale_original_units" in source:
            destination["reference_scale_original_units"] = source["reference_scale_original_units"]
        destination["n_observations"] = destination.get("n_observations", 0) + source.get("n_observations", 0)
        metrics = destination.setdefault("metrics", {})
        for metric_name, metric in source.get("metrics", {}).items():
            merged = metrics.setdefault(metric_name, {})
            for units in ("original_units", "reference_scaled"):
                stat = metric.get(units)
                if stat is None:
                    continue
                total = merged.setdefault(units, {"numerator": 0.0, "denominator": 0.0, "count": 0, "reason": None})
                if stat.get("numerator") is not None:
                    total["numerator"] += stat["numerator"]
                total["denominator"] += stat.get("denominator", 0.0)
                total["count"] += stat.get("count", 0) or 0
                if stat.get("reason") and not total["reason"]:
                    total["reason"] = stat["reason"]
            merged["original_unit"] = metric.get("original_unit")
        source_diag = source.get("diagnostics")
        if source_diag:
            _merge_diagnostics(destination.setdefault("diagnostics", {}), source_diag)

    def add(self, report):
        self.reference_scale_frozen = report.get("reference_scale_frozen", self.reference_scale_frozen)
        self.patient_aggregation_available = report.get(
            "patient_aggregation_available", self.patient_aggregation_available
        )
        for name, signal in report.get("by_signal", {}).items():
            self._merge_signal(self.by_signal.setdefault(name, {}), signal)
        for horizon, signals in report.get("by_horizon_minutes", {}).items():
            horizon_map = self.by_horizon.setdefault(horizon, {})
            for name, signal in signals.items():
                self._merge_signal(horizon_map.setdefault(name, {}), signal)
        for metric_name, by_patient in report.get("patient_statistics", {}).items():
            totals = self.patient_totals.setdefault(metric_name, {})
            for patient_id, stat in by_patient.items():
                current = totals.setdefault(str(patient_id), [0.0, 0])
                if stat.get("numerator") is not None:
                    current[0] += stat["numerator"]
                current[1] += stat.get("denominator", 0)

    @staticmethod
    def _finalize_signal_map(signals):
        result = {}
        for name, source in signals.items():
            signal = {
                "unit": source.get("unit"),
                "n_observations": source.get("n_observations", 0),
                "tail_thresholds": source.get("tail_thresholds", {}),
                "metrics": {},
            }
            if "reference_scale_original_units" in source:
                signal["reference_scale_original_units"] = source["reference_scale_original_units"]
            for metric_name, metric in source.get("metrics", {}).items():
                output = {"original_unit": metric.get("original_unit")}
                for units, values in metric.items():
                    if units not in ("original_units", "reference_scaled"):
                        continue
                    if values["denominator"]:
                        output[units] = _stat(
                            values["numerator"], values["denominator"], count=values["count"]
                        )
                    else:
                        output[units] = _stat(
                            None, 0, reason=values.get("reason") or "no_valid_observations", count=0
                        )
                signal["metrics"][metric_name] = output
            if "diagnostics" in source:
                signal["diagnostics"] = source["diagnostics"]
            result[name] = signal
        return result

    def finalize(self):
        by_signal = self._finalize_signal_map(self.by_signal)
        aggregates = {}
        for metric_name in METRIC_KINDS:
            available = [
                signal["metrics"].get(metric_name, {}).get("reference_scaled")
                for signal in by_signal.values()
            ]
            available = [stat for stat in available if stat and stat["value"] is not None]
            micro_numerator = sum(stat["numerator"] for stat in available)
            micro_denominator = sum(stat["denominator"] for stat in available)
            variable_values = [stat["value"] for stat in available]
            patient_stats = self.patient_totals.get(metric_name, {}).values()
            patient_means = [numerator / denominator for numerator, denominator in patient_stats if denominator]
            aggregates[metric_name] = {
                "micro_per_observation": _stat(
                    micro_numerator if micro_denominator else None,
                    micro_denominator,
                    reason="no_valid_observations",
                ),
                "macro_per_variable": _stat(
                    sum(variable_values) if variable_values else None,
                    len(variable_values),
                    reason="no_variable_with_valid_score",
                ),
                "mean_per_patient": _stat(
                    sum(patient_means) if patient_means else None,
                    len(patient_means),
                    reason="no_patient_with_valid_score",
                ),
            }
        return {
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
            "reference_scale_frozen": bool(self.reference_scale_frozen),
            "patient_aggregation_available": bool(self.patient_aggregation_available),
            "n_observations": sum(signal.get("n_observations", 0) for signal in by_signal.values()),
            "by_signal": by_signal,
            "by_horizon_minutes": {
                horizon: self._finalize_signal_map(signals)
                for horizon, signals in self.by_horizon.items()
            },
            "aggregates": aggregates,
        }


def _merge_diagnostics(destination, source):
    pit = destination.setdefault("pit", {"count": 0, "histogram": [], "rank_histogram": []})
    source_pit = source.get("pit", {})
    pit["count"] += source_pit.get("count", 0)
    pit["histogram"] = _sum_lists(pit["histogram"], source_pit.get("histogram", []))
    pit["rank_histogram"] = _sum_lists(pit["rank_histogram"], source_pit.get("rank_histogram", []))

    reliability = destination.setdefault("event_reliability", {})
    for tail, values in source.get("event_reliability", {}).items():
        current = reliability.setdefault(tail, {
            "threshold": values["threshold"],
            "bins": [{"count": 0, "predicted_probability_sum": 0.0, "observed_event_sum": 0} for _ in values["bins"]],
        })
        for index, bin_value in enumerate(values["bins"]):
            current_bin = current["bins"][index]
            for key in ("count", "predicted_probability_sum", "observed_event_sum"):
                current_bin[key] += bin_value[key]

    query_stats = destination.setdefault("quantiles_by_query_minute", {})
    for query, values in source.get("quantiles_by_query_minute", {}).items():
        current = query_stats.setdefault(query, {
            "count": 0, "observed_sum": 0.0,
            "predicted_quantile_sums": {level: 0.0 for level in values["predicted_quantile_sums"]},
        })
        current["count"] += values["count"]
        current["observed_sum"] += values["observed_sum"]
        for level, total in values["predicted_quantile_sums"].items():
            current["predicted_quantile_sums"][level] += total


def _sum_lists(left, right):
    if not left:
        return list(right)
    if not right:
        return left
    if len(left) != len(right):
        raise ValueError("Diagnostic histogram sizes changed between evaluation batches")
    return [a + b for a, b in zip(left, right)]
