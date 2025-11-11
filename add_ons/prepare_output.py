from typing import Dict, Any, Tuple,Optional,List
import numpy as np
from sklearn.model_selection import train_test_split
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from utils.build_multi_input_dataset import build_multiinput_dataset
from utils.padd_list import padding_list
import torch

class PrepareOutputAddOn(BaseAddOn):
    """
    Transformation add-on that prepares final dataset objects (NumPy arrays or
    PyTorch datasets) from the pipeline's postprocessed state.

    This add-on extracts sequence features (X), labels (y), and sequence lengths 
    from a SequenceCollection and packages them into the format required for 
    training (e.g., NumPy arrays for scikit-learn models or MultiInputDataset 
    for PyTorch). It also handles train/validation splitting and metadata filtering.

    Operates on:
    - state['samples']: SequenceCollection - The core feature and label data.
    - state['y_padded']: Optional padded labels (usually set by LabelPadder).
    """
    def __init__(self, metadata_keys: Optional[List[str]] = None):
        """
        Initializes the PrepareOutputAddOn.

        Args:
            metadata_keys: An optional list of strings specifying which keys 
                           from the sample's metadata dictionary should be 
                           retained and included in the final dataset object 
                           (only applicable when `for_torch=True`). If None, 
                           no metadata is included.
                           Example: ['asset_id', 'sequence_mask']
        """
        super().__init__()
        self.metadata_keys = metadata_keys

    def _filter_metadata(self, metadata_list: List[Dict[str, Any]], keys: Optional[List[str]]) -> Optional[List[Dict[str, Any]]]:
        """
        Filters a list of metadata dictionaries to only keep specified keys.

        Args:
            metadata_list: A list where each element is a dictionary containing 
                           all metadata for a single sample.
            keys: A list of keys (strings) to retain in the metadata dictionaries.

        Returns:
            A new list of dictionaries, where each dictionary only contains the 
            specified keys. Returns `None` if `keys` is empty or None.
        """
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
        Prepares final dataset objects from the pipeline's state, optionally 
        performing a train/validation split.

        If `for_torch=True`, the output is a `MultiInputDataset` (or a pair of 
        datasets if splitting). If `for_torch=False`, the output is a NumPy array
        or a tuple of arrays for splitting.

        Args:
            state: The current pipeline state dictionary, expected to contain 
                   'samples' (SequenceCollection) and potentially 'y_padded'.
            pipeline_extra_info: Additional information from the pipeline.
            val_split: If True, splits the data into training and validation sets.
            test_size: The proportion of the data to use for the validation set 
                       if `val_split` is True (e.g., 0.2 means 20% validation).
            random_state: Seed used by the random number generator for splitting.
            for_torch: If True, prepares PyTorch `MultiInputDataset` objects.
                       If False, prepares flattened NumPy arrays.

        Returns:
            If `for_torch=False`:
                - No split: (X_all, y_padded_all, state)
                - With split: state['final_output'] = (X_train, y_train, X_val, y_val, state)
            
            If `for_torch=True`:
                - No split: state['final_output'] = (dataset_all, state)
                - With split: state['final_output'] = (train_dataset, val_dataset, state)

            In all cases, the primary output is stored in `state["final_output"]`.
        
        Raises:
            TypeError: If `state['samples']` is not a SequenceCollection.
        """
        samples: SequenceCollection = state.get("samples")
        if not isinstance(samples, SequenceCollection):
            raise TypeError("State must contain a 'samples' key with a SequenceCollection.")

        X_list = [s.X for s in samples]
        y_list = [s.y for s in samples]
        x_lengths_list = [s.x_lengths for s in samples]
        # --- 1. Extract and Filter the metadata list ---
        raw_metadata_list = [s.metadata for s in samples]
        # Filter the full list based on required keys (e.g., ['mask', 'asset_id'])
        metadata_list = self._filter_metadata(raw_metadata_list, self.metadata_keys)
        # ----------------------------------------------
        y_padded, max_len_y = padding_list(y_list)

        if not for_torch:
            # NumPy / scikit-learn format (single flattened array)
            X = np.array([x["main"].to_numpy().flatten() for x in X_list])
            if not val_split:
                return X, y_padded, state

            # Perform train/validation split (NumPy)
            idx_train, idx_val = train_test_split(
                np.arange(len(X_list)), test_size=test_size, random_state=random_state
            )
            X_train, X_val = X[idx_train], X[idx_val]
            y_train, y_val = y_padded[idx_train], y_padded[idx_val]
            state["final_output"] = (X_train, y_train, X_val, y_val, state)
            return state
        
        else:
            # Torch / Multi-input dataset format
            if not val_split:
                # Build a single dataset
                dataset = build_multiinput_dataset(
                    X_list, y_padded, x_lengths_list, metadata_list
                )
                state["final_output"] = (dataset, state)
                return state
            else:
                # Perform train/validation split (Torch)
                indices = np.arange(len(X_list))
                idx_train, idx_val = train_test_split(
                    indices, test_size=test_size, random_state=random_state
                )

                # Split feature lists
                X_train = [X_list[i] for i in idx_train]
                X_val = [X_list[i] for i in idx_val]
                
                # Split metadata list
                if metadata_list is not None:
                    metadata_train = [metadata_list[i] for i in idx_train]
                    metadata_val = [metadata_list[i] for i in idx_val]
                else:
                    metadata_train = None
                    metadata_val = None

                # Build train and validation datasets
                train_dataset = build_multiinput_dataset(
                    X_train, 
                    y_padded[idx_train], 
                    [x_lengths_list[i] for i in idx_train],
                    metadata_train
                )
                val_dataset = build_multiinput_dataset(
                    X_val, 
                    y_padded[idx_val], 
                    [x_lengths_list[i] for i in idx_val],
                    metadata_val
                )
                
                state["final_output"] = (train_dataset, val_dataset, state)
                return state
        
    def on_server_request(
        self,
        state: Dict[str, Any],
        pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepares model-ready inputs for inference from the processed state.

        This method is optimized for server-side inference where a model needs 
        a single PyTorch-compatible dataset (MultiInputDataset) built from 
        the processed sequence. Labels (y) are ignored or set to a dummy value.

        Args:
            state: The current pipeline state dictionary, expected to contain 
                   'samples' (SequenceCollection).
            pipeline_extra_info: Additional information from the pipeline.

        Returns:
            A `MultiInputDataset` object ready for a PyTorch DataLoader.

        Raises:
            TypeError: If `state['samples']` is not a SequenceCollection.
        """
        samples: SequenceCollection = state.get("samples")
        if not isinstance(samples, SequenceCollection):
            raise TypeError("State must contain a 'samples' key with a SequenceCollection.")

        # Extract features and lengths
        X_list = [s.X for s in samples]
        x_lengths_list = [s.x_lengths for s in samples]

        # Create dummy y (required for build_multiinput_dataset signature)
        dummy_y = np.zeros(len(X_list), dtype=np.float32)

        # Build the dataset for inference. Metadata is not included here
        # but could be added by passing self.metadata_keys to build_multiinput_dataset
        dataset = build_multiinput_dataset(X_list, dummy_y, x_lengths_list)

        return dataset