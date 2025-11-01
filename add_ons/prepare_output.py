from typing import Dict, Any, Tuple,Optional,List
import numpy as np
from sklearn.model_selection import train_test_split
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from utils.build_multi_input_dataset import build_multiinput_dataset
import torch

class PrepareOutputAddOn(BaseAddOn):
    """
    Transformation add-on that prepares final dataset objects (NumPy arrays or
    PyTorch datasets) from the pipeline's postprocessed state.

    Operates on:
    - state['samples']: SequenceCollection
    - state['y_padded']: Optional padded labels (usually set by LabelPadder)
    """
    def __init__(self, metadata_keys: Optional[List[str]] = None):
            super().__init__()
            self.metadata_keys = metadata_keys

    def _filter_metadata(self, metadata_list: List[Dict[str, Any]], keys: Optional[List[str]]) -> List[Dict[str, Any]]:
            """Filters each dictionary in the list to only keep specified keys."""
            if not keys:
                return None # Return None to let MultiInputDataset handle empty metadata correctly
            
            # Filter each dictionary in the list
            filtered_list = []
            for meta_dict in metadata_list:
                filtered_list.append({k: meta_dict.get(k) for k in keys if k in meta_dict})
                
            return filtered_list


    def on_final_output(
            self,
            state: Dict[str, Any],
            pipeline_extra_info: Dict[str, Any],
            val_split: bool = False,
            test_size: float = 0.2,
            random_state: int = 42,
            for_torch: bool = True,
        ) -> Tuple:
            """
            Prepares final dataset objects from the SequenceCollection.
            ... (docstring omitted for brevity)
            """
            samples: SequenceCollection = state.get("samples")
            if not isinstance(samples, SequenceCollection):
                raise TypeError("State must contain a 'samples' key with a SequenceCollection.")

            X_list = [s.X for s in samples]
            y_list = [s.y for s in samples]
            x_lengths_list = [s.x_lengths for s in samples]
            
            # --- 1. Extract and Filter the metadata list ---
            raw_metadata_list = [s.metadata for s in samples]
            # Filter the full list based on required keys
            metadata_list = self._filter_metadata(raw_metadata_list, self.metadata_keys)
            # ----------------------------------------------
            
            y_padded = state.get("y_padded", np.array(y_list, dtype=np.float32))

            if not for_torch:
                # ... (NumPy logic remains the same)
                X = np.array([x["main"].to_numpy().flatten() for x in X_list])
                if not val_split:
                    return X, y_padded, state

                idx_train, idx_val = train_test_split(
                    np.arange(len(X_list)), test_size=test_size, random_state=random_state
                )
                X_train, X_val = X[idx_train], X[idx_val]
                y_train, y_val = y_padded[idx_train], y_padded[idx_val]
                state["final_output"] = (X_train, y_train, X_val, y_val, state)
                return state
            
            else:
            # Torch / multi-input
                if not val_split:
                    # --- FIX: Pass metadata_list here too (if not splitting) ---
                    dataset = build_multiinput_dataset(X_list, y_padded, x_lengths_list, metadata_list)
                    # -----------------------------------------------------------
                    state["final_output"] = (dataset, state)
                    return state
                else:
                    indices = np.arange(len(X_list))
                    idx_train, idx_val = train_test_split(indices, test_size=test_size, random_state=random_state)

                    X_train = [X_list[i] for i in idx_train]
                    X_val = [X_list[i] for i in idx_val]
                    
                    # Split the metadata list based on indices
                    # Note: metadata_list is already filtered by this point
                    if metadata_list is not None:
                        metadata_train = [metadata_list[i] for i in idx_train]
                        metadata_val = [metadata_list[i] for i in idx_val]
                    else:
                        metadata_train = None
                        metadata_val = None

                    # --- FIX: Pass metadata_train and metadata_val to the builders ---
                    train_dataset = build_multiinput_dataset(
                        X_train, y_padded[idx_train], [x_lengths_list[i] for i in idx_train],
                        metadata_train # <-- Passed here
                    )
                    val_dataset = build_multiinput_dataset(
                        X_val, y_padded[idx_val], [x_lengths_list[i] for i in idx_val],
                        metadata_val # <-- Passed here
                    )
                    # -----------------------------------------------------------------
                    
                    state["final_output"] = (train_dataset, val_dataset, state)
                    return state
        
    def on_server_request(
        self,
        state: Dict[str, Any],
        pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepares model-ready inputs for inference from processed state.
        """
        samples: SequenceCollection = state.get("samples")
        if not isinstance(samples, SequenceCollection):
            raise TypeError("State must contain a 'samples' key with a SequenceCollection.")

        # Extract features and lengths
        X_list = [s.X for s in samples]
        x_lengths_list = [s.x_lengths for s in samples]

        # ✅ Create dummy y (just to satisfy dataset builder)
        dummy_y = np.zeros(len(X_list), dtype=np.float32)

        # ✅ Reuse your existing helper to build a consistent dataset
        dataset = build_multiinput_dataset(X_list, dummy_y, x_lengths_list)

        return dataset