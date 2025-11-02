from typing import Dict, Any
import numpy as np
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection

class ValueExtenderAddOn(BaseAddOn):
    """
    Extends each sample's target (y) array by a fixed number of values.

    Example:
        ValueExtenderAddOn(n=2, v=1)
        - Adds two elements of value 1.0 to the end of each y array.

    Operates on:
    - state['samples']: SequenceCollection
    """

    def __init__(self, n: int, v: float):
        """
        Args:
            n: Number of values to append to each y.
            v: Value to use for the appended elements.
        """
        self.n = n
        self.v = v

    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extends each sample's y array and updates state['samples'] accordingly.
        """
        samples: SequenceCollection = state.get("samples")
        if not isinstance(samples, SequenceCollection):
            raise TypeError("State must contain a 'samples' key with a SequenceCollection.")

        for s in samples:
            if s.y is not None:
                # Ensure numpy array
                y_arr = np.array(s.y, dtype=np.float32)
                extension = np.full(self.n, self.v, dtype=np.float32)
                s.y = np.concatenate([y_arr, extension])
        
        # update state['samples']
        state["samples"] = samples
        return state
