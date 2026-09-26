"""Minute-based observation and forecast windows shared by the pipeline."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

TARGET_NAMES = ('HR', 'SBP', 'RR', 'Temperature', 'O2 Saturation')
REQUIRED_TARGET_NAMES = ('HR', 'SBP', 'RR')

HISTORY_MINUTES = 60
FORECAST_MINUTES = 20
STRIDE_MINUTES = 20
WINDOW_MINUTES = HISTORY_MINUTES + FORECAST_MINUTES
INFO_COLUMNS = ['ts_ind', 'sub_id', 'x_len', 'y_len', 'window_start']
REJECTION_REASONS = (
    'history_count', 'history_coverage', 'history_recency', 'history_gap',
    'forecast_count', 'forecast_coverage', 'nonfinite_values',
)


@dataclass(frozen=True)
class WindowQuality:
    """Coverage thresholds in minute bins, checked before history subsampling."""

    history_min_observations: int = 4
    history_block_minutes: int = 20
    history_max_age_minutes: int = 10
    history_max_gap_minutes: int = 20
    forecast_min_observations: int = 2
    forecast_block_minutes: int = 10

    def __post_init__(self):
        for name, value in vars(self).items():
            duration = HISTORY_MINUTES if name.startswith('history_') else FORECAST_MINUTES
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= duration:
                raise ValueError(f'{name} must be an integer between 1 and {duration}.')
            if 'block_minutes' in name and duration % value:
                raise ValueError(f'{name} must divide {duration} minutes exactly.')

    def rejection_reasons(self, history, forecast):
        """All failed rules for one required signal; times are relative to start."""
        history, forecast = np.unique(history), np.unique(forecast)
        failed = set()
        if len(history) < self.history_min_observations:
            failed.add('history_count')
        history_blocks = np.unique(history // self.history_block_minutes)
        if len(history_blocks) < HISTORY_MINUTES // self.history_block_minutes:
            failed.add('history_coverage')
        if not len(history) or HISTORY_MINUTES - history[-1] > self.history_max_age_minutes:
            failed.add('history_recency')
        if np.any(np.diff(history) > self.history_max_gap_minutes):
            failed.add('history_gap')
        if len(forecast) < self.forecast_min_observations:
            failed.add('forecast_count')
        forecast_blocks = np.unique((forecast - HISTORY_MINUTES) // self.forecast_block_minutes)
        if len(forecast_blocks) < FORECAST_MINUTES // self.forecast_block_minutes:
            failed.add('forecast_coverage')
        return failed


def split_stays(data, num_parts=20):
    """Keep every ICU stay intact when distributing preprocessing work."""
    return [
        data.loc[data.ts_ind.isin(stay_ids)]
        for stay_ids in np.array_split(data.ts_ind.unique(), num_parts)
    ]


def generate_windows(data, target_var, required_var=None, *, quality=WindowQuality(), report=None):
    """Return 60+20 minute windows with per-required-signal coverage, stride 20.

    Input minutes are integer bins relative to ICU admission (step_2).
    History is [start, start+60); targets are [start+60, start+80).
    All targets are required unless required_var explicitly selects a subset.
    Optional targets are retained when observed, without interpolation.
    Only windows whose final minute bin has been reached are considered.
    report, if supplied, is a Counter updated once per window and failed rule;
    reasons overlap, so their counts need not sum to rejected_windows.
    """
    target_var = tuple(target_var)
    required_var = tuple(target_var if required_var is None else required_var)
    if not required_var or not set(required_var) <= set(target_var):
        raise ValueError('required_var must be a nonempty subset of target_var.')
    if report is None:
        report = Counter()
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
            report['candidate_windows'] += 1
            failed = set()
            for variable in required_var:
                history = minutes[left:split][variables[left:split] == variable] - start
                forecast = minutes[split:right][variables[split:right] == variable] - start
                failed.update(quality.rejection_reasons(history, forecast))
            y_indices = np.arange(split, right)[is_target[split:right]]
            indices = np.concatenate((np.arange(left, split), y_indices))
            if not np.isfinite(values[indices]).all():
                failed.add('nonfinite_values')
            if failed:
                report['rejected_windows'] += 1
                report.update(failed)
                continue
            report['accepted_windows'] += 1

            x_len, y_len = split - left, len(y_indices)
            y_mask = np.zeros(len(indices))
            y_mask[x_len:] = 1
            samples.append([
                variables[indices], minutes[indices] - start, values[indices], y_mask
            ])
            info.append([int(stay_id), int(stay.sub_id.iloc[0]), int(x_len), y_len, start])
    return samples, info
