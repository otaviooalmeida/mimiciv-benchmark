import heapq
import pickle
import os
from pathlib import Path
import numpy as np
from preprocess.storage import DATASET_FORMAT, EventStore, map_array, drop_pages
from torch.utils.data import DataLoader, Dataset
from preprocess.windowing import FORECAST_MINUTES, HISTORY_MINUTES, WINDOW_MINUTES

def _select_history_indices(variables, times, size, target_var, recent_per_target):
    """Reserve recent targets, then balance context across observed variables.

    Only historical IDs/timestamps enter this policy. Context quotas favor the
    least-represented variable, breaking ties by most recent available time and
    variable ID. Within each quota, spread indices over the remaining timeline
    (including both ends when quota >= 2; use the latest when quota == 1).
    """
    order = np.argsort(times, kind='stable')
    if len(order) <= size:
        return order

    groups = {}
    for index in order:
        groups.setdefault(int(variables[index]), []).append(index)
    targets = set(target_var)
    selected, candidates, quotas, queue = [], {}, {}, []
    for variable, indices in groups.items():
        reserved = min(recent_per_target, len(indices)) if variable in targets else 0
        if reserved:
            selected.extend(indices[-reserved:])
            indices = indices[:-reserved]
        if indices:
            candidates[variable] = np.asarray(indices)
            quotas[variable] = 0
            heapq.heappush(queue, (reserved, -times[indices[-1]], variable))

    if len(selected) > size:
        raise ValueError(
            f'History budget {size} cannot preserve {len(selected)} recent target observations. '
            'Increase diffusion.size or lower diffusion.recent_per_target.'
        )

    for _ in range(size - len(selected)):
        count, negative_time, variable = heapq.heappop(queue)
        quotas[variable] += 1
        if quotas[variable] < len(candidates[variable]):
            heapq.heappush(queue, (count + 1, negative_time, variable))
    for variable, quota in quotas.items():
        if quota:
            indices = candidates[variable]
            positions = (np.linspace(0, len(indices) - 1, quota, dtype=int)
                         if quota > 1 else np.array([len(indices) - 1]))
            selected.extend(indices[positions])

    keep = np.zeros(len(times), dtype=bool)
    keep[selected] = True
    return order[keep[order]]


def triplet_generate(data, info, size, target_var, recent_per_target=3):
    for name, value in [('size', size), ('recent_per_target', recent_per_target)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    triplets_x = np.zeros((len(data), 4, size))
    triplets_y = np.zeros((len(data), 4, FORECAST_MINUTES * len(target_var)))

    for i in range(len(data)):
        _fill_triplets(data[i], int(info.iloc[i]['x_len']), int(info.iloc[i]['y_len']),
                       triplets_x[i], triplets_y[i], target_var, recent_per_target)
    return triplets_x, triplets_y, info


def _fill_triplets(sample, x_len, y_len, x, y, target_var, recent_per_target):
    """Shared numerical path for legacy in-memory and disk-backed datasets."""
    x_times = np.asarray(sample[1][:x_len])
    y_times = np.asarray(sample[1][x_len:x_len + y_len])
    if (y_len > y.shape[-1]
            or not np.isfinite(x_times).all() or not np.isfinite(y_times).all()
            or np.any((x_times < 0) | (x_times >= HISTORY_MINUTES))
            or np.any((y_times < HISTORY_MINUTES) | (y_times >= WINDOW_MINUTES))):
        raise ValueError(
            f'Invalid {HISTORY_MINUTES}+{FORECAST_MINUTES} minute sample. '
            'Rerun preprocess/step_2.py through step_4.py.'
        )
    selected = _select_history_indices(
        np.asarray(sample[0][:x_len]), x_times, x.shape[-1], target_var, recent_per_target,
    )
    x[3, :len(selected)] = 1
    y[3, :y_len] = 1
    for k in range(3):
        values = np.asarray(sample[k])
        x[k, :len(selected)] = values[selected]
        y[k, :y_len] = values[x_len:x_len + y_len]
                    
class MIMIC_Dataset(Dataset):
    def __init__(self, data, info, size, target_var, use_index_list=None, *, recent_per_target=3):
        if 'window_start' not in info.columns:
            raise ValueError(
                'Outdated dataset. Rerun preprocess/step_2.py through step_4.py '
                f'for {HISTORY_MINUTES}+{FORECAST_MINUTES} minute windows.'
            )
        self.samples_x, self.samples_y, self.info = triplet_generate(
            data, info, size, target_var, recent_per_target=recent_per_target,
        )
        self.info = self.info[['ts_ind', 'x_len', 'y_len', 'window_start']].to_numpy(copy=True)
        self.use_index_list = np.arange(len(self.samples_x))
    
    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "samples_x": self.samples_x[index],
            "samples_y": self.samples_y[index],
            "info": self.info[index]
        }
        
        return s

    def __len__(self):
        return len(self.use_index_list)
        
class DiskDataset(Dataset):
    """Read a window and normalize/select its history only when requested."""
    def __init__(self, data_path, manifest, split, size, target_var, *, recent_per_target=3):
        for name, value in [('size', size), ('recent_per_target', recent_per_target)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        self.base = Path(data_path).resolve().parent
        self.manifest = manifest
        self.split = split
        self.size = size
        self.targets = tuple(map(int, target_var))
        self.recent = recent_per_target
        self.length = manifest['lengths'][('train', 'valid', 'test').index(split)]
        self._events = self._rows = None
        self._pid = None
        self._reads = 0

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        # Never pickle mmap contents when DataLoader workers use spawn.
        state.update(_events=None, _rows=None, _pid=None)
        return state

    def __getitem__(self, index):
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        if self._pid != os.getpid():
            self._events = EventStore(self.base, self.manifest['events'])
            self._rows = map_array(self.base / self.manifest['root'] / (self.split + '.bin'), '<i8', 8)
            if len(self._rows) != self.length:
                raise ValueError('Dataset index length does not match its manifest.')
            self._pid = os.getpid()
        row = self._rows[index]
        sample = self._events.sample(row, self.targets)
        means = self.manifest['means'][sample[0]]
        stds = self.manifest['stds'][sample[0]]
        sample[2] = np.where(np.isnan(means), 0., (sample[2] - means) / np.where(stds == 0, 1., stds))
        x = np.zeros((4, self.size))
        y = np.zeros((4, FORECAST_MINUTES * len(self.targets)))
        _fill_triplets(sample, int(row[2]), int(row[3]), x, y, self.targets, self.recent)
        info = np.asarray(row[[0, 2, 3, 4]]).copy()
        self._reads += 1
        if self._reads % 256 == 0:
            self._events.release_pages()
            drop_pages(self._rows)
        return {'samples_x': x, 'samples_y': y, 'info': info}


def get_dataloader(data_path, var_path, size, batch_size=32, *, recent_per_target=3):
    with open(data_path, 'rb') as file:
        saved = pickle.load(file)
    with open(var_path, 'rb') as file:
        var, target_var = pickle.load(file)
    if isinstance(saved, dict):
        if (saved.get('format') != DATASET_FORMAT or saved['variables'] != list(var)
                or saved['targets'] != list(map(int, target_var))):
            raise ValueError('Dataset metadata does not match var.pkl. Rerun preprocessing.')
        datasets = [DiskDataset(data_path, saved, split, size, target_var, recent_per_target=recent_per_target)
                    for split in ('train', 'valid', 'test')]
    else:
        # Existing checkpoints/fixtures can still use old normalized datasets.
        datasets = [MIMIC_Dataset(saved[i], saved[i + 1], size, target_var, recent_per_target=recent_per_target)
                    for i in (0, 2, 4)]
    return tuple(DataLoader(data, batch_size=batch_size, shuffle=bool(len(data))) for data in datasets)
