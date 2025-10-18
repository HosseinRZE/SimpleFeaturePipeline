from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

def filter_invalid_sequences(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters sequences where the feature data in any of the nested DataFrames/Series 
    within the 'X_list' dictionaries contains NaN values. Also filters the 
    corresponding x_lengths and y_lengths.

    Args:
        state (Dict[str, Any]): The pipeline state dictionary which must contain
                                'X_list', 'y_list', 'x_lengths', and 'y_lengths'.

    Returns:
        Dict[str, Any]: The updated state dictionary with invalid sequences and 
                        lengths removed.
    """
    
    X_list: List[Dict[str, Any]] = state['X_list']
    y_list: List[np.ndarray] = state['y_list']
    x_lengths: List[int] = state['x_lengths']
    y_lengths: List[int] = state['y_lengths']
    
    initial_count = len(X_list)

    # 1. Define the check function for a single sample (which is a dict of DataFrames/Series/Arrays)
    def is_valid_sample(x_sample: Dict[str, Any]) -> bool:
        """Checks if any data component inside the sample dict contains NaN."""
        # Iterate over all feature components (e.g., 'main', 'aux', etc.)
        for component_data in x_sample.values():
            if isinstance(component_data, (pd.DataFrame, pd.Series)):
                # Pandas object check
                if component_data.isnull().values.any():
                    return False
            elif isinstance(component_data, np.ndarray):
                # NumPy array check (for potential future steps)
                if np.isnan(component_data).any():
                    return False
        return True

    # 2. Identify the indices of valid sequences
    valid_indices: List[int] = [
        i for i, x in enumerate(X_list) if is_valid_sample(x)
    ]

    # 3. Filter all lists in one go
    state['X_list'] = [X_list[i] for i in valid_indices]
    state['y_list'] = [y_list[i] for i in valid_indices]
    state['x_lengths'] = [x_lengths[i] for i in valid_indices]
    state['y_lengths'] = [y_lengths[i] for i in valid_indices]

    final_count = len(state['X_list'])
    deleted_count = initial_count - final_count

    # 4. Print deletion status
    if deleted_count > 0:
        print(f"⚠️ Filtered out {deleted_count} sequences due to NaN values (Removed {deleted_count/initial_count:.2%} of total). Remaining sequences: {final_count}")
    else:
        print("✅ No sequences were filtered out (no NaN values found).")

    return state