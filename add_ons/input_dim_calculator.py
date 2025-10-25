from typing import Dict, Any
import pandas as pd
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection

class InputDimCalculator(BaseAddOn):
    """
    Calculates the input dimension (number of features) for each feature group
    based on the first sample in the SequenceCollection.

    ---
    **Purpose**
        Determines how many features (columns) each group in `sample.X` contains.
        This information is used by model components to configure input layers
        for each feature group (e.g., 'main', 'aux').

    ---
    **Input Structure**
        state['samples']: SequenceCollection
            A collection of SequenceSample objects.
            Each SequenceSample has:
                - X[group_key]: pandas.DataFrame of shape (T x F)
                  representing time-series features.

    ---
    **Output Structure**
        state['input_dim']: Dict[str, int]
            Maps each feature group to its feature count (F).

            Example:
                {
                    "main": 32,
                    "aux": 5
                }

    ---
    **Example 1 — Single Feature Group**
        Suppose:
            sample.X = {
                "main": pd.DataFrame(columns=["open", "high", "low", "close"])
            }

        Then:
            state['input_dim'] = {"main": 4}

    ---
    **Example 2 — Multiple Feature Groups**
        Suppose:
            sample.X = {
                "main": pd.DataFrame(columns=["price", "volume", "trend"]),
                "aux": pd.DataFrame(columns=["indicator1", "indicator2"])
            }

        Then:
            state['input_dim'] = {"main": 3, "aux": 2}

    ---
    **Notes**
        - The add-on inspects only the first sample, assuming consistent shapes
          across all samples in the SequenceCollection.
        - Supports both pandas DataFrames and NumPy arrays as feature containers.
    """

    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        samples: SequenceCollection = state.get('samples')
        if samples is None or len(samples) == 0:
            print("⚠️ No samples found in state['samples']. Skipping InputDimCalculator.")
            state['input_dim'] = {}
            return state

        # Take the first sample as representative
        first_sample = next(iter(samples))
        input_dim = {}

        for group_key, group_data in first_sample.X.items():
            if isinstance(group_data, pd.DataFrame):
                input_dim[group_key] = group_data.shape[1]
            elif hasattr(group_data, 'shape') and len(group_data.shape) > 1:
                # Fallback for numpy arrays or tensors
                input_dim[group_key] = group_data.shape[1]
            else:
                print(f"⚠️ Unexpected type for '{group_key}' in InputDimCalculator: {type(group_data)}")

        state['input_dim'] = input_dim
        print(f"✅ Calculated input_dim: {input_dim}")
        return state
