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
    **Purpose**
        Ensures data integrity by removing sequences that contain NaNs in any of 
        their feature groups (e.g., 'main', 'aux'). This prevents runtime errors 
        during model training and improves data quality consistency.

    ---
    **Input**
        state['samples']: SequenceCollection
            Contains multiple SequenceSample objects.
            Each SequenceSample has:
                - X[group_key]: pandas.DataFrame (T x F)
                - y: np.ndarray

    ---
    **Output**
        state['samples']: SequenceCollection
            A filtered version of the collection containing only valid samples.

    ---
    **Validation Criteria**
        A sample is considered *invalid* if **any** of the following is true:
        - Any DataFrame or Series in `sample.X[group_key]` contains NaN values.
        - Any NumPy array in `sample.X[group_key]` contains NaN values.

    ---
    **Example**
        Suppose:
            sample_1.X["main"] = pd.DataFrame([[1, 2], [3, np.nan]])
            sample_2.X["main"] = pd.DataFrame([[1, 2], [3, 4]])

        Then only `sample_2` is kept.

        Result:
            state['samples'] = SequenceCollection([sample_2])
    """

    def apply_window(
        self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        samples: SequenceCollection = state.get("samples")
        if not samples or len(samples) == 0:
            print("⚠️ No samples found in state['samples']. Skipping FilterInvalidSequencesAddOn.")
            return state

        initial_count = len(samples)

        # --- Compact, Pythonic NaN check ---
        def is_valid_sample(sample: SequenceSample) -> bool:
            """Returns True if all feature groups are NaN-free."""
            return all(
                not (
                    (isinstance(data, (pd.DataFrame, pd.Series)) and data.isnull().values.any())
                    or (isinstance(data, np.ndarray) and np.isnan(data).any())
                )
                for data in sample.X.values()
            )

        # --- Filter samples with a single comprehension ---
        valid_samples: List[SequenceSample] = [s for s in samples if is_valid_sample(s)]

        # --- Update state ---
        state["samples"] = SequenceCollection(valid_samples)

        # --- Logging ---
        final_count = len(valid_samples)
        deleted_count = initial_count - final_count
        if deleted_count > 0:
            print(
                f"⚠️ Filtered out {deleted_count} invalid sequences due to NaN values "
                f"({deleted_count / initial_count:.2%} of total). Remaining: {final_count}"
            )
        else:
            print("✅ No invalid sequences found (no NaN values detected).")

        return state
