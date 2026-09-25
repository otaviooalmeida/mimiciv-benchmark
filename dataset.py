import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from preprocess.windowing import FORECAST_MINUTES, HISTORY_MINUTES, WINDOW_MINUTES

def triplet_generate(data, info, size, target_var):
    triplets_x = np.zeros((len(data), 4, size))
    triplets_y = np.zeros((len(data), 4, FORECAST_MINUTES * len(target_var)))
    
    for i in range(len(data)):
        pos = 0
        x_len = int(info.iloc[i]['x_len'])
        y_len = int(info.iloc[i]['y_len'])
        x_times = np.asarray(data[i][1][:x_len])
        y_times = np.asarray(data[i][1][x_len:x_len + y_len])
        if (y_len > triplets_y.shape[-1]
                or np.any((x_times < 0) | (x_times >= HISTORY_MINUTES))
                or np.any((y_times < HISTORY_MINUTES) | (y_times >= WINDOW_MINUTES))):
            raise ValueError('Invalid 60+20 minute sample. Rerun preprocess/step_2.py through step_4.py.')
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
                    s = int(np.random.rand() * x_len)
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
    def __init__(self, data, info, size, target_var, use_index_list=None):
        if 'window_start' not in info.columns:
            raise ValueError('Outdated dataset. Rerun preprocess/step_2.py through step_4.py for 60+20 minute windows.')
        self.samples_x, self.samples_y, self.info = triplet_generate(data, info, size, target_var)
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
        
def get_dataloader(data_path, var_path, size, batch_size=32):
    train_set, train_info, valid_set, valid_info, test_set, test_info = pickle.load(open(data_path, 'rb'))
    var, target_var = pickle.load(open(var_path, 'rb'))
    train_data = MIMIC_Dataset(train_set, train_info, size, target_var)
    valid_data = MIMIC_Dataset(valid_set, valid_info, size, target_var)
    test_data = MIMIC_Dataset(test_set, test_info, size, target_var)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=1)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=1)
    
    return train_loader, valid_loader, test_loader
