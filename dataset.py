import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

def triplet_generate(data, info, size, target_var, rng=None):
    triplets_x = np.zeros((len(data), 4, size))
    triplets_y = np.zeros((len(data), 4, 10 * len(target_var)))
    
    for i in range(len(data)):
        pos = 0
        x_len = info.iloc[i]['x_len']
        y_len = info.iloc[i]['y_len']
        triplets_y[i, 3, :y_len] = 1
        for j in range(y_len):
            for k in range(3):
                triplets_y[i, k, pos] = data[i][k][x_len + j]
            pos += 1
        
        # feature selection
        pos = 0
        if x_len > size:
            out = False
            vs = None
            triplets_x[i, 3] = 1
            ct = pd.DataFrame(np.zeros(len(target_var)).reshape((1, -1)), columns = list(target_var))
            while pos < size:
                if out:
                    out = False
                    ct = ct.drop(columns = vs)
                if len(ct.columns) > 0:
                    vs = ct.columns[-1]
                    for j in range(len(ct.columns) - 1):
                        if ct[ct.columns[j]].item() <= ct[ct.columns[j + 1]].item():
                            vs = ct.columns[j]
                            break
                        
                    out = True
                    for j in range(x_len):
                        if data[i][0][j] == vs:
                            for k in range(3):
                                triplets_x[i, k, pos] = data[i][k][j]
                                data[i][k] = np.delete(data[i][k], j)
                            x_len -= 1
                            pos += 1
                            ct[vs] += 1
                            out = False
                            break
                        
                else:
                    random_value = np.random.rand() if rng is None else rng.random()
                    s = int(random_value * x_len)
                    for k in range(3):
                        triplets_x[i, k, pos] = data[i][k][s]
                        data[i][k] = np.delete(data[i][k], s)
                        
                    x_len -= 1
                    pos += 1
            s = np.argsort(triplets_x[i][1])
            for j in range(3):
                triplets_x[i][j] = triplets_x[i][j][s]
                
        else:
            triplets_x[i, 3, :x_len] = 1
            for j in range(x_len):
                for k in range(3):
                    triplets_x[i, k, pos] = data[i][k][j]
                pos += 1
    
    return triplets_x, triplets_y, info
                    
class MIMIC_Dataset(Dataset):
    def __init__(self, data, info, size, target_var, use_index_list=None, seed=2026):
        rng = np.random.default_rng(seed)
        self.samples_x, self.samples_y, self.info = triplet_generate(data, info, size, target_var, rng=rng)
        if 'sub_id' in self.info.columns:
            patient_ids = self.info['sub_id'].to_numpy()
            sample_info = self.info.drop(columns=['sub_id']).to_numpy()
            # Keep ts_ind/x_len/y_len at their existing positions and append patient ID.
            self.info = np.column_stack((sample_info, patient_ids))
        else:
            self.info = np.asarray(self.info)
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
        
def get_dataloader(data_path, var_path, size, batch_size=32, seed=2026, include_test=True):
    prepared = pickle.load(open(data_path, 'rb'))
    if len(prepared) == 8:
        (train_set, train_info, val_set, val_info,
         calibration_set, calibration_info, test_set, test_info) = prepared
    elif len(prepared) == 6:
        # Legacy datasets remain usable for model validation and explicit test runs.
        train_set, train_info, val_set, val_info, test_set, test_info = prepared
        calibration_set = calibration_info = None
    else:
        raise ValueError("dataset.pkl must contain either legacy 3-split or current 4-split data")
    var, target_var = pickle.load(open(var_path, 'rb'))
    train_data = MIMIC_Dataset(train_set, train_info, size, target_var, seed=seed)
    val_data = MIMIC_Dataset(val_set, val_info, size, target_var, seed=seed + 1)
    calibration_data = None if calibration_set is None else MIMIC_Dataset(
        calibration_set, calibration_info, size, target_var, seed=seed + 2
    )
    test_seed = seed + 3 if calibration_data is not None else seed + 2
    test_data = MIMIC_Dataset(test_set, test_info, size, target_var, seed=test_seed) if include_test else None

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_model_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    calibration_loader = None if calibration_data is None else DataLoader(
        calibration_data, batch_size=batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    test_loader = None if test_data is None else DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(test_seed),
    )
    return train_loader, val_model_loader, calibration_loader, test_loader
