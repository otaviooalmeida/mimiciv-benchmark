import argparse
from collections import Counter, deque
import concurrent.futures
import multiprocessing
from dataclasses import asdict
from functools import partial
import json
import pickle
from pathlib import Path

import yaml

from partition_io import read_partitions
from windowing import (
    FORECAST_MINUTES, HISTORY_MINUTES, REJECTION_REASONS, REQUIRED_TARGET_NAMES,
    STRIDE_MINUTES, TARGET_NAMES, WindowQuality, iter_windows,
)


def sample(data, part_no, *, target_var, required_var, quality, chunk_size):
    report = Counter()
    samples, info, kept_subjects = [], [], set()
    chunk_no = 0

    def flush():
        path = Path('data/first') / f'samples_{part_no + 1:06d}_{chunk_no:06d}.pkl'
        with path.open('wb') as file:
            pickle.dump([samples, info], file, protocol=pickle.HIGHEST_PROTOCOL)
        samples.clear()
        info.clear()

    for window, row in iter_windows(data, target_var, required_var, quality=quality, report=report):
        samples.append(window)
        info.append(row)
        kept_subjects.add(row[1])
        if len(samples) >= chunk_size:
            flush()
            chunk_no += 1
    if samples:
        flush()
    print(f'Part {part_no + 1} finished: {report["accepted_windows"]} windows', flush=True)
    return report, kept_subjects, set(map(int, data.sub_id.unique()))


def process_parts(parts, worker, workers):
    if workers == 1:
        # No child process or IPC copies in the default, lowest-memory mode.
        for part_no, data in enumerate(parts):
            yield worker(data, part_no)
        return

    # Do not use executor.map: on Python 3.12 it eagerly queues the entire input.
    # Spawn also prevents workers from inheriting the parent's input DataFrames.
    parts = enumerate(parts)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
    ) as executor:
        pending = deque()

        def submit_next():
            item = next(parts, None)
            if item is not None:
                part_no, data = item
                pending.append(executor.submit(worker, data, part_no))

        for _ in range(workers):
            submit_next()
        while pending:
            yield pending.popleft().result()
            submit_next()


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select windows with HR, SBP and RR coverage.')
    parser.add_argument('--quality-config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'config/windowing.yaml')
    parser.add_argument('--workers', type=positive_int, default=1,
                        help='Concurrent partitions (default: 1, no multiprocessing).')
    parser.add_argument('--chunk-size', type=positive_int, default=1000,
                        help='Maximum accepted windows per output file (default: 1000).')
    args = parser.parse_args()
    with args.quality_config.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError('Window quality configuration must be a mapping of thresholds.')
    quality = WindowQuality(**config)
    with open('data/var.pkl', 'rb') as file:
        variables, target_var = pickle.load(file)
    if [variables[index] for index in target_var] != list(TARGET_NAMES):
        raise ValueError('Outdated targets. Rerun preprocess/step_2.py for HR, SBP and RR.')
    required_var = [variables.index(name) for name in REQUIRED_TARGET_NAMES]
    Path('data/first').mkdir(parents=True, exist_ok=True)
    Path('data/window_quality_report.json').unlink(missing_ok=True)
    # Remove old worker outputs so a rerun with fewer parts cannot mix cohorts.
    for path in Path('data/first').glob('samples_*.pkl'):
        path.unlink()
    worker = partial(sample, target_var=target_var, required_var=required_var,
                     quality=quality, chunk_size=args.chunk_size)
    counts, kept_subjects, all_subjects = Counter(), set(), set()
    for report, subjects, input_subjects in process_parts(
        read_partitions('data/sets.pkl'), worker, args.workers,
    ):
        counts.update(report)
        kept_subjects.update(subjects)
        all_subjects.update(input_subjects)
    summary = {
        'target_signals': TARGET_NAMES,
        'required_signals': REQUIRED_TARGET_NAMES,
        'quality': asdict(quality),
        'history_minutes': HISTORY_MINUTES,
        'forecast_minutes': FORECAST_MINUTES,
        'stride_minutes': STRIDE_MINUTES,
        **{key: counts[key] for key in ('candidate_windows', 'accepted_windows', 'rejected_windows')},
        'patients_in_input': len(all_subjects),
        'patients_with_accepted_windows': len(kept_subjects),
        'patients_without_accepted_windows': len(all_subjects - kept_subjects),
        'rejections_by_reason': {reason: counts[reason] for reason in REJECTION_REASONS},
        'note': 'Each rejected window can fail multiple rules; reason counts overlap.',
    }
    with open('data/window_quality_report.json', 'w') as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
