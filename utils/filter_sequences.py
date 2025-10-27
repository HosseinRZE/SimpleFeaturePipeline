from typing import Dict, Any, List
import numpy as np
import pandas as pd
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample


class FilterInvalidSequencesAddOn(BaseAddOn):
    """
    Removes invalid samples from the SequenceCollection if any feature group contains NaN values.

    ---
    Purpose:
        Ensures data integrity by removing sequences that contain NaNs in any of
        their feature groups (e.g., 'main', 'aux').
    """

    def apply_window(
        self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        samples: SequenceCollection = state.get("samples")
        if not isinstance(samples, SequenceCollection) or len(samples) == 0:
            return state

        def is_valid_sample(sample: SequenceSample) -> bool:
            """Returns True if all feature groups are NaN-free."""
            return all(
                not (
                    (isinstance(data, (pd.DataFrame, pd.Series)) and data.isnull().values.any())
                    or (isinstance(data, np.ndarray) and np.isnan(data).any())
                )
                for data in sample.X.values()
            )

        valid_samples: List[SequenceSample] = [s for s in samples if is_valid_sample(s)]
        state["samples"] = SequenceCollection(valid_samples)
        return state

    def on_server_request(
        self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Applies the same NaN filtering logic at inference time."""
        return self.apply_window(state, pipeline_extra_info)
