import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Tuple

# Make sure pad_sequence is imported
# from torch.nn.utils.rnn import pad_sequence 

def collate_batch(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, int], Dict[str, Any]]]):
    """
    Pads variable-length sequences in the batch and aggregates per-group lengths.

    Each item in the batch is expected to be a tuple:
        (X_dict: {'main': Tensor(T, F), 'aux': Tensor(T', F'), ...},
         y_tensor: Tensor(target_dim),
         x_lengths_dict: {'main': int, 'aux': int, ...},
         metadata_dict: {'last_close_price': float, ...}
        )
    """

    # --- 1️⃣ Unzip all 4 components ---
    # MODIFIED: Unpack metadata_list as well
    X_list, y_list, x_lengths_list, metadata_list = zip(*batch) 

    # --- 2️⃣ Pad per feature group ---
    X_dict_out = {}
    for group_key in X_list[0].keys():
        group_seqs = [x[group_key] for x in X_list]
        X_dict_out[group_key] = pad_sequence(group_seqs, batch_first=True)  # (B, T_max_group, F_group)

    # --- 3️⃣ Stack targets ---
    y_tensor = torch.stack(y_list)

    # --- 4️⃣ Convert per-group lengths to tensors ---
    lengths_dict_out = {}
    for group_key in x_lengths_list[0].keys():
        group_lengths = [torch.tensor(x[group_key], dtype=torch.long) for x in x_lengths_list]
        lengths_dict_out[group_key] = torch.stack(group_lengths)  # (B,)

    # --- 5️⃣ Return batched structure ---
    # MODIFIED: Return metadata_list (it's just a list of dicts, no padding needed)
    return X_dict_out, y_tensor, lengths_dict_out, metadata_list