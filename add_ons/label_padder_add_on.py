import numpy as np
from typing import Dict, Any
from add_ons.base_addon import BaseAddOn 

class LabelPadder(BaseAddOn):
    """
    An add-on to pad the list of label arrays into a single NumPy matrix.
    
    It calculates the maximum label length across all samples, creates a
    zero-padded matrix, and stores both the matrix ('y_padded') and the
    max length ('max_len_y') in the state.
    """
    def transformation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Applies padding to the entire y_list."""
        y_list = state['y_list']
        y_lengths = state['y_lengths']

        # Pad labels to max length
        max_len_y = max(y_lengths, default=0)
        y_padded = np.zeros((len(y_list), max_len_y), dtype=np.float32)
        for i, arr in enumerate(y_list):
            # This logic correctly handles cases where labels might have been set to 0
            # and only pads the truly non-zero values.
            nonzero = arr[arr != 0]
            y_padded[i, :len(nonzero)] = nonzero
        
        # Add the new, padded array and max length to the state
        state['y_padded'] = y_padded
        state['max_len_y'] = max_len_y
        
        print(f"✅ Padded labels to shape: {y_padded.shape}")
        return state