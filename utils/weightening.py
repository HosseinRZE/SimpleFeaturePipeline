import numpy as np
# --- Compute label weights ---
def get_label_weights(y_encoded, mlb, label_weighting=None):
    n_labels = y_encoded.shape[1]
    weights = np.ones(n_labels)

    if label_weighting is None or label_weighting == "none":
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

    raise ValueError("label_weighting must be None, 'none', dict, or 'scale_pos'")