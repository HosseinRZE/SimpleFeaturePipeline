import torch
from torch.utils.data import Dataset
import pandas as pd
class MultiInputDataset(Dataset):
    def __init__(self, X_dict, y, x_lengths,metadata_list=None):
        self.X_dict = X_dict
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = len(y) if y is not None else 0
        self.metadata_list = metadata_list if metadata_list is not None else [{} for _ in range(self.length)]
        # 🧩 Keep dicts as-is — do NOT wrap in torch.tensor
        self.x_lengths = x_lengths  
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.X_dict.items():
            feature_data = v[idx]
            if isinstance(feature_data, (pd.DataFrame, pd.Series)):
                tensor_data = feature_data.values
            else:
                tensor_data = feature_data
            sample[k] = torch.tensor(tensor_data, dtype=torch.float32)

        return sample, self.y[idx], self.x_lengths[idx], self.metadata_list[idx]