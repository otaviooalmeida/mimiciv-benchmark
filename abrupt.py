import numpy as np


def abrupt_window_flags(samples, training_indices, target_var):
    """Flag windows with a >training-p95 target change rate during the horizon."""
    target_var = np.asarray(target_var)
    training_indices = set(training_indices)
    rates = []
    training_rates = {int(variable): [] for variable in target_var}

    for index, sample in enumerate(samples):
        vind, minute, value = sample[:3]
        window_rates = {}
        for variable in target_var:
            positions = np.flatnonzero(vind == variable)
            positions = positions[np.argsort(np.asarray(minute)[positions], kind="stable")]
            variable_rates = []
            for left, right in zip(positions[:-1], positions[1:]):
                elapsed = minute[right] - minute[left]
                if 1 <= elapsed <= 10 and 30 <= minute[right] < 40:
                    rate = abs(value[right] - value[left]) / elapsed
                    if np.isfinite(rate):
                        variable_rates.append(float(rate))
            window_rates[int(variable)] = variable_rates
            if index in training_indices:
                training_rates[int(variable)].extend(variable_rates)
        rates.append(window_rates)

    thresholds = {
        variable: float(np.percentile(values, 95))
        for variable, values in training_rates.items() if values
    }
    flags = np.array([
        any(rate > thresholds.get(variable, np.inf)
            for variable, values in window.items() for rate in values)
        for window in rates
    ], dtype=np.int64)
    return flags, thresholds
