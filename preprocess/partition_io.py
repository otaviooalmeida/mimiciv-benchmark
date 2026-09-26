"""Read one stay-preserving input partition at a time, including legacy inputs."""
from pathlib import Path
import pickle
import tempfile
import warnings

import numpy as np

FORMAT = {'format': 'tdstf-partition-stream-v1'}
ROWS_PER_PART = 250_000


def iter_stay_parts(data, rows_per_part=ROWS_PER_PART):
    """Target a row budget without ever splitting an ICU stay.

    A single stay larger than the budget is emitted alone. Only the current
    partition is copied; unlike split_stays(), no list of DataFrames is built.
    """
    if rows_per_part < 1:
        raise ValueError('rows_per_part must be positive.')
    indices, count = [], 0
    for stay_indices in data.groupby('ts_ind', sort=False).indices.values():
        if indices and count + len(stay_indices) > rows_per_part:
            yield data.iloc[np.concatenate(indices)]
            indices, count = [], 0
        indices.append(stay_indices)
        count += len(stay_indices)
    if indices:
        yield data.iloc[np.concatenate(indices)]


def write_partitions(parts, path):
    """Atomically replace a stream; independent pickle records bound reader RAM."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            pickle.dump(FORMAT, file, protocol=pickle.HIGHEST_PROTOCOL)
            for part in parts:
                if not part.empty:
                    pickle.dump(part, file, protocol=pickle.HIGHEST_PROTOCOL)
            # Explicit terminator distinguishes completion from truncated data.
            pickle.dump(None, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _upgrade_legacy(path):
    # This load is unavoidable for the old single-pickle list. Convert before
    # starting workers, then release the list before reading the new stream.
    with Path(path).open('rb') as file:
        first = pickle.load(file)
    if isinstance(first, dict) and first == FORMAT:
        return
    if not isinstance(first, list):
        raise ValueError('Unknown sets.pkl format. Rerun preprocess/step_2.py.')
    warnings.warn(
        'Converting legacy sets.pkl to bounded partitions. This one-time conversion '
        'loads the old list into RAM and requires space for a second copy on disk. '
        'If conversion runs out of memory, rerun preprocess/step_2.py.',
        stacklevel=2,
    )

    def parts():
        for i in range(len(first)):
            part = first[i]
            first[i] = None
            yield from iter_stay_parts(part)

    write_partitions(parts(), path)


def read_partitions(path):
    """Yield independent partitions, upgrading old sets.pkl atomically once."""
    _upgrade_legacy(path)
    with Path(path).open('rb') as file:
        header = pickle.load(file)
        if header != FORMAT:
            raise ValueError('Unknown partition stream format.')
        while True:
            try:
                part = pickle.load(file)
            except EOFError as error:
                raise ValueError('Truncated sets.pkl. Rerun preprocess/step_2.py.') from error
            if part is None:
                return
            yield part
