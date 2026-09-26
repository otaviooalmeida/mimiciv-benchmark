import argparse
from collections import Counter, deque
import concurrent.futures
import multiprocessing
from dataclasses import asdict
from functools import partial
import json
import os
import pickle
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from storage import (WINDOW_FORMAT, EventStore, atomic_pickle, write_event_store)
from windowing import (
    FORECAST_MINUTES, HISTORY_MINUTES, REJECTION_REASONS, REQUIRED_TARGET_NAMES,
    STRIDE_MINUTES, TARGET_NAMES, WindowQuality, iter_window_indices,
)


def sample(bounds, part_no, *, data_dir, events_descriptor, output_root,
           target_var, required_var, quality, chunk_size):
    events = EventStore(data_dir, events_descriptor)
    report, kept_subjects, input_subjects = Counter(), set(), set()
    batch = []
    with (Path(output_root) / f'windows_{part_no:06d}.bin').open('wb') as file:
        for stay_id, subject, begin, end in events.stays[bounds[0]:bounds[1]]:
            input_subjects.add(int(subject))
            for start, left, split, right, future in iter_window_indices(
                events.minutes[begin:end], events.variables[begin:end], events.values[begin:end],
                target_var, required_var, quality=quality, report=report,
                release_pages=events.release_pages,
            ):
                batch.append([stay_id, subject, split - left, len(future), start,
                              begin + left, begin + split, begin + right])
                kept_subjects.add(int(subject))
                if len(batch) >= chunk_size:
                    np.asarray(batch, dtype='<i8').tofile(file)
                    batch.clear()
                    events.release_pages()
            events.release_pages()
        if batch:
            np.asarray(batch, dtype='<i8').tofile(file)
    print(f'Part {part_no + 1} finished: {report["accepted_windows"]} windows', flush=True)
    return report, kept_subjects, input_subjects


def process_parts(parts, worker, workers):
    if workers == 1:
        for part_no, data in enumerate(parts):
            yield worker(data, part_no)
        return
    # At most workers small offset tasks, never DataFrames or event arrays.
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
                        help='Concurrent offset tasks (default: 1, no multiprocessing).')
    parser.add_argument('--chunk-size', type=positive_int, default=1000,
                        help='Maximum window index rows buffered before writing (default: 1000).')
    parser.add_argument('--legacy-input', action='store_true',
                        help='Explicitly convert old sets.pkl. May load the entire old input into RAM; '
                             'prefer rerunning the updated step_2.py.')
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
    try:
        events = EventStore.open('data/sets.pkl')
    except ValueError:
        if not args.legacy_input:
            raise
        from partition_io import read_partitions
        write_event_store(read_partitions('data/sets.pkl'), 'data/sets.pkl')
        events = EventStore.open('data/sets.pkl')

    first = Path('data/first')
    first.mkdir(parents=True, exist_ok=True)
    (first / '.incomplete').touch()
    (first / 'manifest.pkl').unlink(missing_ok=True)
    Path('data/window_quality_report.json').unlink(missing_ok=True)
    # Old sample chunks must not be mistaken for the new run, including empty runs.
    for path in first.glob('samples_*.pkl'):
        path.unlink()
    output_root = Path(tempfile.mkdtemp(prefix='windows_', dir='data'))
    worker = partial(sample, data_dir='data', events_descriptor=events.descriptor('data'),
                     output_root=output_root, target_var=target_var, required_var=required_var,
                     quality=quality, chunk_size=args.chunk_size)
    counts, kept_subjects, all_subjects = Counter(), set(), set()
    try:
        # Fixed-size ranges of stay metadata, regardless of event/window counts.
        parts = ((i, min(i + 32, len(events.stays))) for i in range(0, len(events.stays), 32))
        for report, subjects, input_subjects in process_parts(parts, worker, args.workers):
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
        atomic_pickle({
            'format': WINDOW_FORMAT, 'root': os.path.relpath(output_root, first),
            'events': events.descriptor(first), 'variables': list(variables),
            'targets': list(map(int, target_var)), 'quality': asdict(quality),
        }, first / 'manifest.pkl')
        (first / '.incomplete').unlink()
    except BaseException:
        shutil.rmtree(output_root)
        raise
    print(json.dumps(summary, indent=2))
