import argparse
from collections import Counter
import concurrent.futures
from dataclasses import asdict
from functools import partial
import json
import pickle
from pathlib import Path

import yaml

from windowing import (
    FORECAST_MINUTES, HISTORY_MINUTES, REJECTION_REASONS, REQUIRED_TARGET_NAMES,
    STRIDE_MINUTES, TARGET_NAMES, WindowQuality, generate_windows,
)


def sample(data, thread, *, target_var, required_var, quality):
    report = Counter()
    samples, info = generate_windows(data, target_var, required_var, quality=quality, report=report)
    with open('data/first/samples_{}.pkl'.format(thread + 1), 'wb') as file:
        pickle.dump([samples, info], file)
    print('Thread_{} finished'.format(thread))
    return report, {row[1] for row in info}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select windows with HR, SBP and RR coverage.')
    parser.add_argument('--quality-config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'config/windowing.yaml')
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
    with open('data/sets.pkl', 'rb') as file:
        sets = pickle.load(file)
    Path('data/first').mkdir(parents=True, exist_ok=True)
    # Remove old worker outputs so a rerun with fewer parts cannot mix cohorts.
    for path in Path('data/first').glob('samples_*.pkl'):
        path.unlink()
    worker = partial(sample, target_var=target_var, required_var=required_var, quality=quality)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Consume results so worker failures are not silently ignored.
        results = list(executor.map(worker, sets, range(len(sets))))
    counts, kept_subjects = Counter(), set()
    for report, subjects in results:
        counts.update(report)
        kept_subjects.update(subjects)
    all_subjects = {int(subject) for part in sets for subject in part.sub_id.unique()}
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
