import torch
from torch.utils.data import Dataset
import pandas as pd
class MultiInputDataset(Dataset):
    def __init__(self, X_dict, y, x_lengths):
        self.X_dict = X_dict
        self.y = torch.tensor(y, dtype=torch.float32)
        self.x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        self.length = len(y) if y is not None else 0 # Handle empty case

    def __len__(self):
            return self.length

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.X_dict.items():
            feature_data = v[idx]
            
            # **The Fix:** Convert Pandas object to NumPy array before Tensor conversion.
            if isinstance(feature_data, (pd.DataFrame, pd.Series)):
                tensor_data = feature_data.values 
            else:
                # Assume it's already a NumPy array or list of numbers
                tensor_data = feature_data
                
            sample[k] = torch.tensor(tensor_data, dtype=torch.float32)

        return sample, self.y[idx], self.x_lengths[idx]