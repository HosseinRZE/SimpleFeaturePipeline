from typing import Dict, Any, List
import numpy as np
import pandas as pd
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample


class FilterInvalidSequencesAddOn(BaseAddOn):
    """
    Removes invalid samples from the SequenceCollection if any feature group contains NaN values.
    Includes logging to track how many samples are filtered.
    ---
    Purpose:
        Ensures data integrity by removing sequences that contain NaNs in any of
        their feature groups (e.g., 'main', 'aux').
    """
    
    # Run this filter later, after feature engineering add-ons
    on_evaluation_priority = 50 

    def apply_window(
        self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        self.add_trace_print(pipeline_extra_info, "Starting 'FilterInvalidSequencesAddOn'...")
        
        samples: SequenceCollection = state.get("samples")
        if not isinstance(samples, SequenceCollection) or len(samples) == 0:
            self.add_trace_print(pipeline_extra_info, "Skipped: 'samples' collection is empty or missing.")
            return state

        initial_count = len(samples)
        self.add_trace_print(pipeline_extra_info, f"Found {initial_count} samples to check for NaNs.")

        def is_valid_sample(sample_idx: int, sample: SequenceSample) -> bool:
            """Returns True if all feature groups are NaN-free."""
            for group_name, data in sample.X.items():
                has_nan = False
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    has_nan = data.isnull().values.any()
                elif isinstance(data, np.ndarray):
                    has_nan = np.isnan(data).any()
                
                if has_nan:
                    # Log the *first* feature group that causes the sample to fail
                    self.add_trace_print(
                        pipeline_extra_info, 
                        f"[Sample {sample_idx}] REJECTED due to NaN in feature group: '{group_name}'."
                    )
                    return False
            
            return True # No NaNs found in any group

        # We use enumerate to get the original index for logging
        valid_samples: List[SequenceSample] = []
        for i, s in enumerate(samples):
            if is_valid_sample(i, s):
                valid_samples.append(s)
        
        final_count = len(valid_samples)
        removed_count = initial_count - final_count

        self.add_trace_print(
            pipeline_extra_info, 
            f"Filtering complete. Initial: {initial_count}, Removed: {removed_count}, Final: {final_count}."
        )

        state["samples"] = SequenceCollection(valid_samples)
        return state

    def on_server_request(
        self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Applies the same NaN filtering logic at inference time."""
        return self.apply_window(state, pipeline_extra_info)