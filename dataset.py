import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

INTERVENTION_NAMES = {
    'Intubated', 'FiO2', 'Norepinephrine', 'Vasopressin', 'Dopamine', 'Epinephrine',
    'Neosynephrine', 'Milrinone', 'Propofol', 'Midazolam', 'Fentanyl', 'Normal Saline',
    'Lactated Ringers', 'Packed RBC', 'Fresh Frozen Plasma', 'Albumin 5%', 'Albumin 25%',
    'OR/PACU Crystalloid',
}


def triplet_generate(data, info, size, target_var, rng=None, intervention_var=()):
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
            ct = pd.DataFrame(np.zeros(len(target_var)).reshape((1, -1)), columns=list(target_var))
            recent = []
            for v in target_var:
                recent.extend(np.flatnonzero(data[i][0][:x_len] == v)[-3:])
            for v in intervention_var:
                recent.extend(np.flatnonzero(data[i][0][:x_len] == v)[-1:])
            recent = np.unique(recent).astype(int)[-size:]
            for j in recent:
                for k in range(3):
                    triplets_x[i, k, pos] = data[i][k][j]
                if data[i][0][j] in target_var:
                    ct[data[i][0][j]] += 1
                pos += 1
            for k in range(3):
                data[i][k] = np.delete(data[i][k], recent)
            x_len -= len(recent)
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
    def __init__(self, data, info, size, target_var, use_index_list=None, seed=2026,
                 intervention_var=()):
        rng = np.random.default_rng(seed)
        self.samples_x, self.samples_y, self.info = triplet_generate(
            data, info, size, target_var, rng=rng, intervention_var=intervention_var,
        )
        self.info = np.array(self.info.drop(columns=['sub_id']))
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
        
def get_dataloader(data_path, var_path, size, batch_size=32, seed=2026):
    train_set, train_info, valid_set, valid_info, test_set, test_info = pickle.load(open(data_path, 'rb'))
    var, target_var = pickle.load(open(var_path, 'rb'))
    intervention_var = [i for i, name in enumerate(var) if name in INTERVENTION_NAMES]
    train_data = MIMIC_Dataset(
        train_set, train_info, size, target_var, seed=seed, intervention_var=intervention_var,
    )
    valid_data = MIMIC_Dataset(
        valid_set, valid_info, size, target_var, seed=seed + 1, intervention_var=intervention_var,
    )
    test_data = MIMIC_Dataset(
        test_set, test_info, size, target_var, seed=seed + 2, intervention_var=intervention_var,
    )

    train_generator = torch.Generator().manual_seed(seed)
    valid_generator = torch.Generator().manual_seed(seed + 1)
    test_generator = torch.Generator().manual_seed(seed + 2)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=train_generator
    )
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, generator=valid_generator
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, generator=test_generator
    )
    
    return train_loader, valid_loader, test_loader
