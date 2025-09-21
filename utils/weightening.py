import numpy as np

def get_label_weights(y_encoded, mlb, label_weighting=None):
    """
    Compute per-label weights for multi-label classification.

    Parameters
    ----------
    y_encoded : np.ndarray
        Binary-encoded labels (n_samples, n_labels)
    mlb : MultiLabelBinarizer
        Used to get class names if dict-based weighting
    label_weighting : None, 'none', dict, 'scale_pos', or False
        - None / 'none' / False: all weights = 1 (no weighting)
        - dict: mapping {label_name: weight}
        - 'scale_pos': inverse frequency weighting

    Returns
    -------
    weights : np.ndarray
        Array of shape (n_labels,) with weights for each label
    """
    n_labels = y_encoded.shape[1]
    weights = np.ones(n_labels)

    if label_weighting in (None, "none", False):
        return weights

    if isinstance(label_weighting, dict):
        for i, label_name in enumerate(mlb.classes_):
            weights[i] = label_weighting.get(label_name, 1)
        return weights

    if label_weighting == "scale_pos":
        for i in range(n_labels):
            n_pos = y_encoded[:, i].sum()
            n_neg = y_encoded.shape[0] - n_pos
            weights[i] = n_neg / n_pos if n_pos > 0 else 1
        return weights

    raise ValueError("label_weighting must be None, 'none', dict, 'scale_pos', or False")
