"""Minute-based observation and forecast windows shared by the pipeline."""

import numpy as np

HISTORY_MINUTES = 60
FORECAST_MINUTES = 20
STRIDE_MINUTES = 20
WINDOW_MINUTES = HISTORY_MINUTES + FORECAST_MINUTES
INFO_COLUMNS = ['ts_ind', 'sub_id', 'x_len', 'y_len', 'window_start']


def split_stays(data, num_parts=20):
    """Keep every ICU stay intact when distributing preprocessing work."""
    return [
        data.loc[data.ts_ind.isin(stay_ids)]
        for stay_ids in np.array_split(data.ts_ind.unique(), num_parts)
    ]


def generate_windows(data, target_var):
    """Return all valid 60+20 minute windows, advancing by 20 minutes.

    Input minutes are integer bins relative to ICU admission (step_2).
    History is [start, start+60); targets are [start+60, start+80).
    No interpolation is performed. As before, require nonempty history and
    at least two observations of one target variable in the forecast interval.
    Only windows whose final minute bin has been reached are considered.
    """
    samples, info = [], []
    for stay_id, stay in data.groupby('ts_ind', sort=False):
        stay = stay.sort_values('minute')
        minutes = stay.minute.to_numpy()
        variables = stay.vind.to_numpy(dtype=np.int64)
        values = stay.value.to_numpy(dtype=float)
        is_target = np.isin(variables, target_var)
        last_start = int(minutes[-1]) + 1 - WINDOW_MINUTES
        for start in range(0, last_start + 1, STRIDE_MINUTES):
            left, split, right = np.searchsorted(
                minutes, [start, start + HISTORY_MINUTES, start + WINDOW_MINUTES]
            )
            if left == split:
                continue
            y_indices = np.arange(split, right)[is_target[split:right]]
            _, counts = np.unique(variables[y_indices], return_counts=True)
            if not np.any(counts > 1):
                continue

            x_len, y_len = split - left, len(y_indices)
            indices = np.concatenate((np.arange(left, split), y_indices))
            y_mask = np.zeros(len(indices))
            y_mask[x_len:] = 1
            samples.append([
                variables[indices], minutes[indices] - start, values[indices], y_mask
            ])
            info.append([int(stay_id), int(stay.sub_id.iloc[0]), int(x_len), y_len, start])
    return samples, info
