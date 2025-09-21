import numpy as np

def pad_sequences_dicts(X_dicts_list, dict_name="main", strategy="forward_fill", pad_value=0.0):
    """
    Pads sequences in X_dicts_list for a given dict_name to the maximum sequence length.

    Parameters:
    -----------
    X_dicts_list : list of dicts
        Each dict contains arrays, e.g., {"main": (seq_len, num_features), ...}
    dict_name : str
        Key of the dict to pad (default "main").
    strategy : str
        Padding strategy. Options:
        - "zeros"        : pad with 0
        - "forward_fill" : pad by repeating last row
        - "constant"     : pad with `pad_value`
        - "mean"         : pad with mean of sequence along axis 0
    pad_value : float
        Value used if strategy="constant"

    Returns:
    --------
    X_padded : np.ndarray
        3D array of shape (n_samples, max_seq_len, num_features)
    """

    # Determine maximum sequence length
    max_len = max(arr[dict_name].shape[0] for arr in X_dicts_list)
    n_features = X_dicts_list[0][dict_name].shape[1]

    X_padded = []
    for arr in X_dicts_list:
        seq = arr[dict_name]
        pad_len = max_len - seq.shape[0]

        if pad_len > 0:
            if strategy == "zeros":
                pad = np.zeros((pad_len, n_features), dtype=np.float32)
            elif strategy == "forward_fill":
                last_row = seq[-1:]
                pad = np.repeat(last_row, pad_len, axis=0)
            elif strategy == "constant":
                pad = np.full((pad_len, n_features), pad_value, dtype=np.float32)
            elif strategy == "mean":
                mean_row = seq.mean(axis=0, keepdims=True)
                pad = np.repeat(mean_row, pad_len, axis=0)
            else:
                raise ValueError(f"Unknown padding strategy '{strategy}'")
            seq = np.vstack([seq, pad])

        X_padded.append(seq)

    return np.array(X_padded, dtype=np.float32)
