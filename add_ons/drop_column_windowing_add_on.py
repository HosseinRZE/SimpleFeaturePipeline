import pandas as pd
from typing import Dict, Any, List
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample

class DropColumnsAddOn(BaseAddOn):
    """
    Removes specified columns from feature DataFrames in each `SequenceSample`
    inside the `SequenceCollection` stored in the pipeline state.
    Works at both training (apply_window) and inference (on_server_request).
    """

    def __init__(self, cols_map: Dict[str, List[str]]):
        super().__init__()
        self.cols_map = cols_map

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Removes unwanted columns from each feature group in the windowed samples."""
        samples: SequenceCollection = state.get('samples')
        if not isinstance(samples, SequenceCollection) or len(samples) == 0:
            return state

        updated_samples: List[SequenceSample] = []

        for sample in samples:
            new_X = {}
            for group_key, df in sample.X.items():
                if not isinstance(df, pd.DataFrame):
                    new_X[group_key] = df
                    continue

                if group_key in self.cols_map:
                    cols_to_drop = self.cols_map[group_key]
                    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

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

        state['samples'] = SequenceCollection(updated_samples)
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference-time hook: apply the same column-dropping logic as during training.
        """
        return self.apply_window(state, pipeline_extra_info)
