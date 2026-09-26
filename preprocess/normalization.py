"""Prepare split indices and normalization without retaining all windows."""
from collections import OrderedDict
from contextlib import closing
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import tempfile

import numpy as np

try:
    from .storage import (DATASET_FORMAT, WINDOW_FORMAT, EventStore, atomic_pickle,
                          drop_pages, import_legacy_windows, map_array, read_manifest)
except ImportError:  # Direct execution of the preprocessing scripts.
    from storage import (DATASET_FORMAT, WINDOW_FORMAT, EventStore, atomic_pickle,
                         drop_pages, import_legacy_windows, map_array, read_manifest)


class StatisticsWriter:
    """Bounded file-handle/buffer cache for the old per-variable rec sequences."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir()
        self.files = OrderedDict()

    def __enter__(self):
        return self

    def append(self, variable, values):
        file = self.files.pop(variable, None)
        if file is None:
            if len(self.files) >= 64:
                _, oldest = self.files.popitem(last=False)
                oldest.close()
            file = (self.root / f'{variable}.bin').open('ab')
        self.files[variable] = file
        file.write(np.asarray(values, dtype='<f8').tobytes())

    def __exit__(self, *args):
        for file in self.files.values():
            file.close()


def numpy_moments(values):
    """Bounded equivalent of NumPy's contiguous float64 mean/std (ddof=0).

    Keep NumPy's buffered reduction order, not Welford/chunk-mean merging. Even
    tiny rounding differences in a near-zero std can materially change the
    normalized benchmark. Parity tests cover buffer boundaries and constants.
    """
    if not len(values):
        return np.nan, np.nan
    block = np.getbufsize()

    def reduce_squared(mean=None):
        total = np.float64(0)
        for start in range(0, len(values), block):
            chunk = values[start:start + block]
            if mean is not None:
                chunk = np.asarray(chunk) - mean
                chunk *= chunk
            total += chunk.sum(dtype=np.float64)
            drop_pages(values)
        return total

    mean = reduce_squared() / len(values)
    std = np.sqrt(reduce_squared(mean) / len(values))
    return mean, std


def build_dataset(data_dir='data', seed=None):
    data_dir = Path(data_dir)
    with (data_dir / 'var.pkl').open('rb') as file:
        variables, targets = pickle.load(file)
    root = Path(tempfile.mkdtemp(prefix='dataset_', dir=data_dir))
    database = root / 'index.sqlite'
    try:
        first = data_dir / 'first'
        if (first / '.incomplete').exists():
            raise ValueError('Step 3 did not finish. Rerun step_3.py before step_4.py.')
        if (first / 'manifest.pkl').exists():
            manifest = read_manifest(first / 'manifest.pkl', WINDOW_FORMAT)
            if manifest['variables'] != list(variables) or manifest['targets'] != list(map(int, targets)):
                raise ValueError('Window metadata does not match var.pkl. Rerun step_3.py.')
            events = EventStore(first, manifest['events'])
            paths = sorted((first / manifest['root']).glob('windows_*.bin'))
        else:
            events = import_legacy_windows(first, root)
            paths = [root / 'windows.bin']

        with closing(sqlite3.connect(database)) as db:
            # SQLite is only a bounded, on-disk grouping/sorting index. No new
            # dependency and no replacement of the benchmark's numerical code.
            db.execute('PRAGMA cache_size=-8192')
            db.execute('PRAGMA temp_store=FILE')
            db.execute('CREATE TABLE windows (id INTEGER PRIMARY KEY, stay INTEGER, '
                       'subject INTEGER, xlen INTEGER, ylen INTEGER, start INTEGER, '
                       'l INTEGER, m INTEGER, r INTEGER)')
            count = 0
            for path in paths:
                rows = map_array(path, '<i8', 8)
                for start in range(0, len(rows), 2048):
                    batch = rows[start:start + 2048].tolist()
                    db.executemany('INSERT INTO windows VALUES (NULL,?,?,?,?,?,?,?,?)', batch)
                    count += len(batch)
                    drop_pages(rows)
            if not count:
                raise ValueError('No windows passed selection. Inspect data/window_quality_report.json before continuing.')
            db.execute('CREATE INDEX by_subject ON windows(subject, id)')
            db.commit()
            subjects = np.array([row[0] for row in db.execute(
                'SELECT DISTINCT subject FROM windows ORDER BY subject')], dtype=np.int64)
            # Same sorted subjects, shuffle algorithm and 64/16/20 cut points.
            # By default keep the old global RNG; an explicit seed is optional.
            rng = np.random if seed is None else np.random.RandomState(seed)
            rng.shuffle(subjects)
            groups = np.split(subjects, [int(.64 * len(subjects)), int(.8 * len(subjects))])
            select = ('SELECT stay,subject,xlen,ylen,start,l,m,r FROM windows '
                      'WHERE subject=? ORDER BY id')

            statistics = root / 'statistics'
            processed = 0
            # Deliberately include validation, and count every overlapping-window
            # history again, in the original subject/window/observation order.
            with StatisticsWriter(statistics) as writer:
                for subject in np.concatenate(groups[:2]):
                    for row in db.execute(select, (int(subject),)):
                        left, split = row[5:7]
                        ids, values = events.variables[left:split], events.values[left:split]
                        for variable in np.unique(ids):
                            writer.append(int(variable), values[ids == variable])
                        processed += 1
                        if processed % 256 == 0:
                            events.release_pages()
            means = np.full(len(variables), np.nan)
            stds = np.full(len(variables), np.nan)
            for variable in range(len(variables)):
                path = statistics / f'{variable}.bin'
                if path.exists():
                    means[variable], stds[variable] = numpy_moments(map_array(path, '<f8'))
            shutil.rmtree(statistics)

            lengths = []
            for name, group in zip(('train', 'valid', 'test'), groups):
                length, batch = 0, []
                with (root / (name + '.bin')).open('wb') as file:
                    for subject in group:
                        for row in db.execute(select, (int(subject),)):
                            batch.append(row)
                            length += 1
                            if len(batch) == 2048:
                                np.asarray(batch, dtype='<i8').tofile(file)
                                batch.clear()
                    if batch:
                        np.asarray(batch, dtype='<i8').tofile(file)
                lengths.append(length)
        database.unlink()
        # Pin the exact input snapshot and statistics, not mutable sets.pkl.
        manifest = {
            'format': DATASET_FORMAT, 'root': os.path.relpath(root, data_dir),
            'events': events.descriptor(data_dir),
            'variables': list(variables), 'targets': list(map(int, targets)),
            'lengths': lengths, 'means': means, 'stds': stds,
            'seed': seed, 'normalization': 'train+validation histories, window-weighted, ddof=0',
        }
        atomic_pickle([means, stds], data_dir / 'mean_std.pkl')
        atomic_pickle(manifest, data_dir / 'dataset.pkl')
        return manifest
    except BaseException:
        shutil.rmtree(root)
        raise
