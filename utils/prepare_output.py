import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Tuple
from utils.build_multi_input_dataset import build_multiinput_dataset

def _prepare_output(state: Dict[str, Any], val_split: bool, test_size: float, random_state: int, for_torch: bool) -> Tuple:
    """Pads, splits, and formats the data for final output."""
    X_list, y_list = state['X_list'], state['y_list']
    x_lengths, y_lengths = state['x_lengths'], state['y_lengths']

    # Flatten X_list to a single numpy array if not for torch
    if not for_torch:
        # Assumes transformations have already created a uniform feature vector per sample
        X = np.array([sample['main'].flatten() for sample in X_list])
    
    # Pad labels to max length
    max_len_y = max(y_lengths, default=0)
    y_padded = np.zeros((len(y_list), max_len_y), dtype=np.float32)
    for i, arr in enumerate(y_list):
        nonzero = arr[arr != 0]
        y_padded[i, :len(nonzero)] = nonzero
    
    # Finalize state params for return
    state['max_len_y'] = max_len_y
    
    if not val_split:
        if for_torch:
            dataset = build_multiinput_dataset(X_list, y_padded, x_lengths)
            return dataset, state
        return X, y_padded, state
    
    # Perform validation split
    indices = np.arange(len(y_list))
    idx_train, idx_val = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    if for_torch:
        X_train = [X_list[i] for i in idx_train]
        X_val = [X_list[i] for i in idx_val]
        train_dataset = build_multiinput_dataset(X_train, y_padded[idx_train], [x_lengths[i] for i in idx_train])
        val_dataset = build_multiinput_dataset(X_val, y_padded[idx_val], [x_lengths[i] for i in idx_val])
        return train_dataset, val_dataset, state

    # For NumPy/XGBoost output
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y_padded[idx_train], y_padded[idx_val]
    return X_train, y_train, X_val, y_val, state