import pandas as pd
from typing import Dict, Any, List
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample

class DropColumnsAddOn(BaseAddOn):
    """
    Removes specified columns from feature DataFrames in each `SequenceSample` inside
    the `SequenceCollection` stored in the pipeline state.

    This add-on operates at the *window level*, i.e., after the data has been segmented
    into time windows but before feature transformations or model input preparation.

    ---
    **Input**
    - `state['samples']`: `SequenceCollection`
        Contains `SequenceSample` objects each with an `X` dict of DataFrames.

    **Config**
    - `self.cols_map`: `Dict[str, List[str]]`
        Mapping of feature group names to columns that should be dropped.
        Example:
            ```python
            {
                "main": ["timestamp", "raw_volume"],
                "aux": ["noise_col"]
            }
            ```

    **Output**
    - `state['samples']`: `SequenceCollection`
        Updated collection where specified columns have been removed from
        the DataFrames in each sample.

    ---
    **Example**
    ```python
    addon = DropColumnsAddOn(cols_map={"main": ["timestamp", "raw_volume"]})
    state = addon.apply_window(state)
    ```
    """

    def __init__(self, cols_map: Dict[str, List[str]]):
        super().__init__()
        self.cols_map = cols_map

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Removes unwanted columns from each feature group in the windowed samples."""
        samples: SequenceCollection = state.get('samples')
        if not isinstance(samples, SequenceCollection) or len(samples) == 0:
            print("⚠️ No valid samples found in state['samples']. Skipping DropColumnsAddOn.")
            return state

        updated_samples: List[SequenceSample] = []
        total_dropped = 0

        for sample in samples:
            new_X = {}

            for group_key, df in sample.X.items():
                if not isinstance(df, pd.DataFrame):
                    new_X[group_key] = df
                    continue

                if group_key in self.cols_map:
                    cols_to_drop = self.cols_map[group_key]
                    cols_to_remove = [c for c in cols_to_drop if c in df.columns]
                    if cols_to_remove:
                        df = df.drop(columns=cols_to_remove)
                        total_dropped += len(cols_to_remove)
                new_X[group_key] = df

            # Preserve all metadata
            updated_samples.append(
                SequenceSample(
                    original_index=sample.original_index,
                    X_features=new_X,
                    y_labels=sample.y,
                    metadata=sample.metadata,
                )
            )

        # Replace old SequenceCollection with updated one
        state['samples'] = SequenceCollection(updated_samples)

        print(f"✅ DropColumnsAddOn: Completed. Total columns dropped across all samples: {total_dropped}")
        return state
