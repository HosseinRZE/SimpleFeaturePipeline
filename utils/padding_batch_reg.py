from torch.nn.utils.rnn import pad_sequence
import torch

from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    """
    Pads variable-length sequences in the batch.
    Supports both single-tensor and dict-of-tensors features.
    """
    Xs, ys = zip(*batch)

    if isinstance(Xs[0], dict):
        # Dict of feature groups
        collated_X = {}
        lengths = None
        for key in Xs[0].keys():
            seqs = [x[key] for x in Xs]  # list of tensors
            lengths = [s.size(0) for s in seqs]  # same for all groups
            collated_X[key] = pad_sequence(seqs, batch_first=True)
    else:
        # Single tensor features
        lengths = [x.size(0) for x in Xs]
        collated_X = pad_sequence(Xs, batch_first=True)

    ys = torch.stack(ys)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return collated_X, ys, lengths

