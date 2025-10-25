from torch.nn.utils.rnn import pad_sequence
import torch

def collate_batch(batch):
    """
    Pads variable-length sequences in the batch and aggregates per-group lengths.

    Each item in the batch is expected to be a tuple:
        (X_dict, y_tensor, x_lengths_dict)
    where:
        - X_dict: {'main': Tensor(T, F), 'aux': Tensor(T', F'), ...}
        - y_tensor: Tensor(target_dim)
        - x_lengths_dict: {'main': int, 'aux': int, ...}
    """

    X_list, y_list, x_lengths_list = zip(*batch)  # unzip batch

    # --- 1️⃣ Pad per feature group ---
    X_dict_out = {}
    for group_key in X_list[0].keys():
        group_seqs = [x[group_key] for x in X_list]
        X_dict_out[group_key] = pad_sequence(group_seqs, batch_first=True)  # (B, T_max_group, F_group)

    # --- 2️⃣ Stack targets ---
    y_tensor = torch.stack(y_list)

    # --- 3️⃣ Convert per-group lengths to tensors ---
    lengths_dict_out = {}
    for group_key in x_lengths_list[0].keys():
        group_lengths = [torch.tensor(x[group_key], dtype=torch.long) for x in x_lengths_list]
        lengths_dict_out[group_key] = torch.stack(group_lengths)  # (B,)

    # --- 4️⃣ Return batched structure ---
    return X_dict_out, y_tensor, lengths_dict_out
