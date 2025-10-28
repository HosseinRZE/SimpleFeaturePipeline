from typing import Dict, Any, Tuple
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

    def on_final_output(
        self,
        state: Dict[str, Any],
        pipeline_extra_info: Dict[str, Any],
        val_split: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
        for_torch: bool = True
    ) -> Tuple:
        """
        Prepares final dataset objects from the SequenceCollection.

        Parameters
        ----------
        state : dict
            Pipeline state containing 'samples' and optionally 'y_padded'.
        pipeline_extra_info : dict
            Extra configuration passed from the pipeline.
        val_split : bool
            Whether to return train/validation split.
        test_size : float
            Fraction of data for validation.
        random_state : int
            Seed for reproducibility of splits.
        for_torch : bool
            Whether to return PyTorch datasets (multi-input) or NumPy arrays.

        Returns
        -------
        Tuple
            Depending on val_split and for_torch, returns either:
            - X, y, state
            - X_train, y_train, X_val, y_val, state
            - dataset, state
            - train_dataset, val_dataset, state
        """
        samples: SequenceCollection = state.get("samples")
        if not isinstance(samples, SequenceCollection):
            raise TypeError("State must contain a 'samples' key with a SequenceCollection.")

        X_list = [s.X for s in samples]
        y_list = [s.y for s in samples]
        x_lengths_list = [s.x_lengths for s in samples]

        y_padded = state.get("y_padded", np.array(y_list, dtype=np.float32))

        if not for_torch:
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
                dataset = build_multiinput_dataset(X_list, y_padded, x_lengths_list)
                state["final_output"] = (dataset, state)
                return state
            else:
                indices = np.arange(len(X_list))
                idx_train, idx_val = train_test_split(indices, test_size=test_size, random_state=random_state)

                X_train = [X_list[i] for i in idx_train]
                X_val = [X_list[i] for i in idx_val]
                train_dataset = build_multiinput_dataset(
                    X_train, y_padded[idx_train], [x_lengths_list[i] for i in idx_train]
                )
                val_dataset = build_multiinput_dataset(
                    X_val, y_padded[idx_val], [x_lengths_list[i] for i in idx_val]
                )
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