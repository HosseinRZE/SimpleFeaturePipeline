from torch.nn.utils.rnn import pad_sequence
import torch
# ---------------- Collate fn for variable-length sequences ---------------- #
def collate_batch(batch):
    """
    Pads variable-length sequences in the batch.

    Args:
        batch: list of tuples (X, y), where
               - X: Tensor of shape (seq_len, feature_dim)
               - y: scalar Tensor (label)
    Returns:
        padded_X: Tensor of shape (batch, max_seq_len, feature_dim)
        y: Tensor of shape (batch,)
        lengths: list of original sequence lengths
    """
    Xs, ys = zip(*batch)
    lengths = [x.size(0) for x in Xs]
    padded_X = pad_sequence(Xs, batch_first=True)  # (batch, max_seq_len, feature_dim)
    y = torch.stack(ys)
    return padded_X, y, torch.tensor(lengths)