import numpy as np
from typing import Dict, Any, List
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection

class LabelPadder(BaseAddOn):
    """
    Pads all label arrays (sample.y) in the SequenceCollection into a single NumPy matrix.

    Input:
        state['samples']: SequenceCollection
            Holds multiple SequenceSample objects with sample.y arrays of variable length.

    Output:
        state['y_padded']: np.ndarray (N x max_len_y)
            Zero-padded 2D array of labels.
        state['max_len_y']: int
            The maximum label length across all samples.
    """

    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        samples: SequenceCollection = state.get('samples')
        if samples is None or len(samples) == 0:
            print("⚠️ No samples found in state['samples']. Skipping LabelPadder.")
            return state

        # Extract y arrays and compute lengths
        y_list: List[np.ndarray] = [sample.y for sample in samples]
        y_lengths = [len(y) for y in y_list]
        max_len_y = max(y_lengths, default=0)

        # Initialize padded matrix
        y_padded = np.zeros((len(y_list), max_len_y), dtype=np.float32)

        # Fill padded array
        for i, arr in enumerate(y_list):
            if arr is None or len(arr) == 0:
                continue
            nonzero = arr[arr != 0]
            y_padded[i, :len(nonzero)] = nonzero

        # Save results to state
        state['y_padded'] = y_padded

        print(f"✅ Padded {len(y_list)} label sequences to shape: {y_padded.shape}")
        return state
