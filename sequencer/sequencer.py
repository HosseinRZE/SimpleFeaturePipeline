from typing import List, Dict, Any
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_sample import SequenceSample
from data_structure.sequence_collection import SequenceCollection
import numpy as np
import pandas as pd


class SequencerAddOn(BaseAddOn):
    """
    Wraps the original create_sequences_by_time function as an add-on.
    Produces `state['samples']` from 'df_data' and 'df_labels'.
    """

    on_evaluation_priority = 0  # runs before all feature add-ons

    def __init__(self, include_cols: List[str] = None, exclude_cols: List[str] = None):
        self.include_cols = include_cols
        self.exclude_cols = exclude_cols

    def sequence_on_train(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]):
        return self._run_sequencer(state, server_mode=False)

    def sequence_on_server(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]):
        return self._run_sequencer(state, server_mode=True)

    def _run_sequencer(self, state: Dict[str, Any], server_mode: bool = False) -> Dict[str, Any]:
        """
        Sequencer logic, unchanged for training.
        For server_mode=True, y_labels are filled with zeros because true labels are unavailable.
        """
        df_data: pd.DataFrame = state['df_data']
        df_labels: pd.DataFrame = state.get('df_labels') if not server_mode else None
        
        include_cols = self.include_cols or []
        exclude_cols = self.exclude_cols or []

        samples: List[SequenceSample] = []
        feature_cols: List[str] = []

        # Determine label columns if training
        if df_labels is not None:
            lineprice_cols = state.get('lineprice_cols', [c for c in df_labels.columns if c.startswith("linePrice")])
        else:
            lineprice_cols = []

        # Iterate over sequences
        if server_mode:
            # During inference, create a single "fake" label row per df_data slice
            # For simplicity, we use the last seq_len rows
            seq_len = state.get("seq_len", 21)
            window_df = df_data.tail(seq_len).copy()
            if not feature_cols:
                all_potential_cols = [c for c in window_df.columns]
                cols_after_exclude = [c for c in all_potential_cols if c not in exclude_cols and c != "timestamp"]
                if include_cols:
                    feature_cols = [c for c in cols_after_exclude if c in include_cols]
                else:
                    feature_cols = cols_after_exclude

            X_df = window_df[feature_cols].astype(np.float32)
            X_dict = {"main": X_df}
            y_labels = np.zeros(len(lineprice_cols) if lineprice_cols else 1, dtype=np.float32)

            sample = SequenceSample(original_index=-1, X_features=X_dict, y_labels=y_labels, metadata={})
            samples.append(sample)

        else:
            # Training mode: use original df_labels
            for original_index, row in df_labels.iterrows():
                mask = (df_data["timestamp"] >= row["startTime"]) & (df_data["timestamp"] <= row["endTime"])
                df_sequence = df_data.loc[mask].copy()
                if df_sequence.empty:
                    continue

                if not feature_cols:
                    all_potential_cols = [c for c in df_sequence.columns]
                    cols_after_exclude = [c for c in all_potential_cols if c not in exclude_cols and c != "timestamp"]
                    if include_cols:
                        feature_cols = [c for c in cols_after_exclude if c in include_cols]
                    else:
                        feature_cols = cols_after_exclude

                X_df = df_sequence[feature_cols].astype(np.float32)
                X_dict = {"main": X_df}
                y_labels = row[lineprice_cols].astype(np.float32).fillna(0).values

                sample = SequenceSample(original_index=original_index, X_features=X_dict, y_labels=y_labels, metadata={})
                samples.append(sample)

        state["samples"] = SequenceCollection(samples)
        return state
