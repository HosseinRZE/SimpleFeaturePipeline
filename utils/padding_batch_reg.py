from torch.nn.utils.rnn import pad_sequence
import torch

def collate_batch(batch):
    """
    Pads variable-length sequences in the batch.

    Args:
        batch: list of tuples (X, y), where
               - X: Tensor of shape (seq_len, feature_dim)
               - y: Tensor of shape (seq_len_y,) or (max_len_y,) with line prices
    Returns:
        padded_X: Tensor of shape (batch, max_seq_len, feature_dim)
        padded_y: Tensor of shape (batch, max_len_y)
        lengths:  Tensor of original input sequence lengths
    """
    Xs, ys = zip(*batch)
    lengths = [x.size(0) for x in Xs]

    # Pad input sequences
    padded_X = pad_sequence(Xs, batch_first=True)  # (batch, max_seq_len, feature_dim)

    # Pad output (line price sequences)
    padded_y = pad_sequence(ys, batch_first=True)  # (batch, max_len_y)

    return padded_X, padded_y, torch.tensor(lengths)
