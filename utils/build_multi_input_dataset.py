import numpy as np
from typing import List, Dict, Any, Union
from utils.multi_input_dataset import MultiInputDataset

def build_multiinput_dataset(
    X_list: List[Dict[str, Any]],
    y_padded: np.ndarray,
    x_lengths: List[Dict[str, int]]
) -> MultiInputDataset:
    """
    Transforms the list of feature dictionaries into a dictionary of feature lists 
    and instantiates the MultiInputDataset.

    ---
    Args:
        X_list:
            List of dictionaries, where each dict represents one sample.
            Example:
                [
                    {'main': df_main_1, 'aux': df_aux_1},
                    {'main': df_main_2, 'aux': df_aux_2},
                    ...
                ]

        y_padded:
            NumPy array of padded target values.

        x_lengths:
            List of dictionaries containing per-group sequence lengths.
            Example:
                [
                    {'main': 120, 'aux': 60},
                    {'main': 128, 'aux': 64},
                    ...
                ]

    ---
    Returns:
        MultiInputDataset
            Dataset ready for PyTorch DataLoader consumption.

    ---
    Notes:
        - Each key in X_dict corresponds to a feature group ('main', 'aux', etc.).
        - x_lengths is preserved as-is (list of dicts) since multi-group models
          may require variable-length handling per feature group.
    """
    if not X_list:
        return MultiInputDataset({}, y_padded, x_lengths)

    # 1. Extract feature group keys from the first sample
    feature_keys = X_list[0].keys()

    # 2. Transpose List[Dict] → Dict[List]
    X_dict = {key: [sample[key] for sample in X_list] for key in feature_keys}

    # 3. Return dataset instance
    return MultiInputDataset(X_dict, y_padded, x_lengths)
