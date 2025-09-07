from torch.nn.utils.rnn import pad_sequence
import torch

def collate_batch(batch):
    """
    Pads variable-length sequences in the batch and returns true lengths.
    """

    X_list, y_list, lengths = zip(*batch)

    # Collect per feature group
    X_dict_out = {}
    for k in X_list[0].keys():
        seqs = [x[k] for x in X_list]
        X_dict_out[k] = pad_sequence(seqs, batch_first=True)  # (B, T_max, F)

    y_tensor = torch.stack(y_list)
    lengths_tensor = torch.stack(lengths)

    return X_dict_out, y_tensor, lengths_tensor


