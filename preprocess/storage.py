"""Disk-backed benchmark storage. Events are stored once; windows are offsets.

All large arrays are raw, fixed-width NumPy files opened read-only with mmap.
Manifests are small, versioned pickles; a completed snapshot is published last.
"""
from contextlib import ExitStack
import mmap
import os
from pathlib import Path
import pickle
import shutil
import tempfile

import numpy as np

EVENT_FORMAT = 'tdstf-events-v1'
WINDOW_FORMAT = 'tdstf-windows-v1'
DATASET_FORMAT = 'tdstf-dataset-v1'
# ts_ind, sub_id, x_len, y_len, window_start, left, split, right
WINDOW_WIDTH = 8


def atomic_pickle(value, path):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_manifest(path, expected_format):
    # Never deserialize a legacy list of DataFrames merely to detect its format.
    with Path(path).open('rb') as file:
        prefix = file.read(4 * 1024 * 1024)
    try:
        manifest = pickle.loads(prefix)
    except (EOFError, pickle.UnpicklingError):
        manifest = None
    if not isinstance(manifest, dict) or manifest.get('format') != expected_format:
        raise ValueError(
            f'{path} is not a {expected_format} manifest. Rerun preprocess/step_2.py '
            'with the updated code. Legacy sets.pkl cannot be loaded partially; '
            'step_3.py --legacy-input explicitly opts into a RAM-intensive conversion.'
        )
    return manifest


def map_array(path, dtype, width=1):
    path = Path(path)
    dtype = np.dtype(dtype)
    size = path.stat().st_size
    if size % (dtype.itemsize * width):
        raise ValueError(f'Truncated array: {path}')
    shape = (size // dtype.itemsize,) if width == 1 else (size // dtype.itemsize // width, width)
    return np.memmap(path, dtype=dtype, mode='r', shape=shape) if size else np.empty(shape, dtype=dtype)


def drop_pages(*arrays):
    # mmap pages are reclaimable, but discard this process's old mappings too so
    # repeated scans do not inflate RSS up to the size of the entire dataset.
    for array in arrays:
        mapping = getattr(array, '_mmap', None)
        if mapping is not None and hasattr(mapping, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
            mapping.madvise(mmap.MADV_DONTNEED)


class EventWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.count = 0

    def __enter__(self):
        self.stack = ExitStack()
        self.files = [self.stack.enter_context((self.root / name).open('wb'))
                      for name in ('variables.bin', 'minutes.bin', 'values.bin', 'stays.bin')]
        return self

    def append(self, variables, minutes, values):
        left = self.count
        for file, values_, dtype in zip(self.files, (variables, minutes, values), ('<i8', '<f8', '<f8')):
            np.asarray(values_, dtype=dtype).tofile(file)
        self.count += len(variables)
        return left, self.count

    def stay(self, stay_id, subject, left, right):
        np.asarray([stay_id, subject, left, right], dtype='<i8').tofile(self.files[3])

    def __exit__(self, *args):
        self.stack.close()


def event_descriptor(root, relative_to):
    return {'format': EVENT_FORMAT, 'root': os.path.relpath(root, relative_to)}


def write_event_store(parts, path):
    """Export step-2 tables, preserving the old per-stay Pandas sort semantics.

    Call from step 2, where the source tables already exist, not by loading a
    giant legacy pickle in step 3. Input parts must contain complete ICU stays.
    """
    path = Path(path)
    root = Path(tempfile.mkdtemp(prefix='events_', dir=path.parent))
    try:
        with EventWriter(root) as writer:
            for part in parts:
                # indices avoids Pandas materializing a reordered copy of all groups.
                for stay_id, indices in part.groupby('ts_ind', sort=False).indices.items():
                    stay = part.iloc[indices].sort_values('minute')
                    left, right = writer.append(stay.vind, stay.minute, stay.value)
                    writer.stay(int(stay_id), int(stay.sub_id.iloc[0]), left, right)
        atomic_pickle(event_descriptor(root, path.parent), path)
    except BaseException:
        shutil.rmtree(root)
        raise


class EventStore:
    def __init__(self, base, descriptor):
        if descriptor.get('format') != EVENT_FORMAT:
            raise ValueError('Unsupported event store format.')
        self.root = Path(base) / descriptor['root']
        self.variables = map_array(self.root / 'variables.bin', '<i8')
        self.minutes = map_array(self.root / 'minutes.bin', '<f8')
        self.values = map_array(self.root / 'values.bin', '<f8')
        self.stays = map_array(self.root / 'stays.bin', '<i8', 4)
        if not len(self.variables) == len(self.minutes) == len(self.values):
            raise ValueError('Inconsistent event array lengths.')

    @classmethod
    def open(cls, path):
        return cls(Path(path).parent, read_manifest(path, EVENT_FORMAT))

    def descriptor(self, base):
        return event_descriptor(self.root, base)

    def sample(self, row, targets):
        _, _, x_len, y_len, start, left, split, right = map(int, row)
        future = np.flatnonzero(np.isin(self.variables[split:right], targets)) + split
        if split - left != x_len or len(future) != y_len:
            raise ValueError('Window index does not match events/target metadata. Rerun preprocessing.')
        indices = np.concatenate((np.arange(left, split), future))
        mask = np.zeros(len(indices))
        mask[x_len:] = 1
        return [self.variables[indices], self.minutes[indices] - start, self.values[indices], mask]

    def release_pages(self):
        drop_pages(self.variables, self.minutes, self.values, self.stays)


def import_legacy_windows(first, destination):
    """One old output chunk at a time; never concatenate the entire cohort.

    Old windows already duplicated their events. This compatibility path cannot
    recover shared offsets, but still allows bounded step-4 and training RAM.
    """
    first, destination = Path(first), Path(destination)
    with EventWriter(destination) as writer, (destination / 'windows.bin').open('wb') as index:
        # Keep os.listdir ordering: it was the ordering used by the old step 4.
        for name in os.listdir(first):
            if not (name.startswith('samples_') and name.endswith('.pkl')):
                continue
            with (first / name).open('rb') as file:
                samples, rows = pickle.load(file)
            if len(samples) != len(rows):
                raise ValueError(f'Mismatched samples/metadata in {name}. Rerun step_3.py.')
            for sample, row in zip(samples, rows):
                stay, subject, x_len, y_len, start = map(int, row)
                if any(len(array) != x_len + y_len for array in sample):
                    raise ValueError(f'Invalid sample lengths in {name}. Rerun step_3.py.')
                left, right = writer.append(sample[0], np.asarray(sample[1]) + start, sample[2])
                np.asarray([stay, subject, x_len, y_len, start, left, left + x_len, right],
                           dtype='<i8').tofile(index)
    return EventStore(destination.parent, event_descriptor(destination, destination.parent))
