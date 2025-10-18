import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Tuple
from utils.build_multi_input_dataset import build_multiinput_dataset

def _prepare_output(state: Dict[str, Any], val_split: bool, test_size: float, random_state: int, for_torch: bool) -> Tuple:
    """Splits and formats the already processed data for final output."""
    X_list = state['X_list']
    x_lengths = state['x_lengths']
    
    # --- ADDED ---
    # Get the padded labels, which were created by the LabelPadder add-on.
    y_padded = state['y_padded']

    # --- DELETED ---
    # The entire padding logic has been moved to the LabelPadder add-on.
    
    # Flatten X_list to a single numpy array if not for torch
    if not for_torch:
        X = np.array([sample['main'].flatten() for sample in X_list])
    
    if not val_split:
        if for_torch:
            dataset = build_multiinput_dataset(X_list, y_padded, x_lengths)
            return dataset, state
        return X, y_padded, state
    
    # Perform validation split
    # --- CHANGED ---
    # The number of samples is now based on the length of X_list or y_padded.
    indices = np.arange(len(X_list))
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