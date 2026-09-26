import heapq
import pickle
import numpy as np
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
        x_len = int(info.iloc[i]['x_len'])
        y_len = int(info.iloc[i]['y_len'])
        x_times = np.asarray(data[i][1][:x_len])
        y_times = np.asarray(data[i][1][x_len:x_len + y_len])
        if (y_len > triplets_y.shape[-1]
                or not np.isfinite(x_times).all() or not np.isfinite(y_times).all()
                or np.any((x_times < 0) | (x_times >= HISTORY_MINUTES))
                or np.any((y_times < HISTORY_MINUTES) | (y_times >= WINDOW_MINUTES))):
            raise ValueError('Invalid 60+20 minute sample. Rerun preprocess/step_2.py through step_4.py.')
        selected = _select_history_indices(
            np.asarray(data[i][0][:x_len]), x_times, size, target_var, recent_per_target,
        )
        triplets_x[i, 3, :len(selected)] = 1
        triplets_y[i, 3, :y_len] = 1
        for k in range(3):
            values = np.asarray(data[i][k])
            triplets_x[i, k, :len(selected)] = values[selected]
            triplets_y[i, k, :y_len] = values[x_len:x_len + y_len]

    return triplets_x, triplets_y, info
                    
class MIMIC_Dataset(Dataset):
    def __init__(self, data, info, size, target_var, use_index_list=None, *, recent_per_target=3):
        if 'window_start' not in info.columns:
            raise ValueError('Outdated dataset. Rerun preprocess/step_2.py through step_4.py for 60+20 minute windows.')
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
        
def get_dataloader(data_path, var_path, size, batch_size=32, *, recent_per_target=3):
    train_set, train_info, valid_set, valid_info, test_set, test_info = pickle.load(open(data_path, 'rb'))
    var, target_var = pickle.load(open(var_path, 'rb'))
    train_data = MIMIC_Dataset(train_set, train_info, size, target_var, recent_per_target=recent_per_target)
    valid_data = MIMIC_Dataset(valid_set, valid_info, size, target_var, recent_per_target=recent_per_target)
    test_data = MIMIC_Dataset(test_set, test_info, size, target_var, recent_per_target=recent_per_target)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=1)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=1)
    
    return train_loader, valid_loader, test_loader
