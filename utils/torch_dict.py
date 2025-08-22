from torch.utils.data import Dataset
class MultiInputDataset(Dataset):
    def __init__(self, data_dict, labels):
        """
        data_dict: dict of {feature_group_name: tensor of shape (num_samples, seq_len, feature_dim)}
        labels: tensor of shape (num_samples,)
        """
        self.data_dict = data_dict
        self.labels = labels
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a dictionary per sample
        sample = {k: v[idx] for k, v in self.data_dict.items()}
        return sample, self.labels[idx]