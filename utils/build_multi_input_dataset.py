import numpy as np
from typing import List, Dict, Any
from utils.multi_input_dataset import MultiInputDataset

def build_multiinput_dataset(X_list: List[Dict[str, Any]], y_padded: np.ndarray, x_lengths: List[int]):
    """
    Transforms the list of feature dictionaries into a dictionary of feature lists 
    and instantiates the MultiInputDataset.

    Args:
        X_list: List of dictionaries, where each dict is a sample. 
                e.g., [{'main': df1, 'aux': df_a1}, {'main': df2, 'aux': df_a2}, ...]
        y_padded: Padded labels (NumPy array).
        x_lengths: True lengths of the sequences.

    Returns:
        MultiInputDataset instance.
    """
    if not X_list:
        return MultiInputDataset({}, y_padded, x_lengths)

    # 1. Get the keys (e.g., 'main', 'aux') from the first sample
    feature_keys = X_list[0].keys()
    
    # 2. Transpose the data: List[Dict] -> Dict[List]
    X_dict = {key: [sample[key] for sample in X_list] for key in feature_keys}
    
    # The X_dict values are now List[DataFrame/Series]

    return MultiInputDataset(X_dict, y_padded, x_lengths)