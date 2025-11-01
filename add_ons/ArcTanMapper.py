import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
# Assuming these imports exist in your environment
from add_ons.base_addon import BaseAddOn 
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample

# --- Constants for Key Names in pipeline_extra_info ---
# We store 'A' in extra_info so it's persisted for inference/evaluation
A_KEY = 'arctan_A_factor' 

class ArctanMapperAddOn(BaseAddOn):
    """
    Applies the transformation f(x) = (2/pi) * arctan(A * (x - 1)) to data.
    
    This function expands values near x=1 aggressively (controlled by A) and 
    clamps the output strictly between -1 and 1. It also handles the inverse 
    transformation for evaluation and inference.

    Args:
        A (float): The tunable scaling factor inside the arctan function.
        **kwargs: Configuration mapping feature groups to column lists, 
                  or 'y' to True/False.
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs
        
        # 1. Extract the tunable 'A' factor
        self.A = self.config.pop('A', 1.0)
        
        # 2. Pre-compile a set of feature group keys for efficient lookup later
        self._target_feature_groups = {
            key for key, val in self.config.items() if key != 'y' and val is not None
        }

    # ----------------------------------------------------------------------
    # --- CORE TRANSFORMATION FUNCTIONS ---
    # ----------------------------------------------------------------------

    def _transform_func(self, x: np.ndarray) -> np.ndarray:
        """Applies the core f(x) = (2/pi) * arctan(A * (x - 1)) transformation."""
        x_float = x.astype(float)
        # (2 / np.pi) is the normalization factor to map the output to (-1, 1)
        return (2.0 / np.pi) * np.arctan(self.A * (x_float - 1.0))

    def _inverse_transform(self, f_x: np.ndarray, A: float) -> np.ndarray:
        """
        Calculates the inverse transformation: x = 1 + (1/A) * tan((pi/2) * f(x)).
        Converts the normalized value f(x) back to the original value x.
        """
        if A is None or A <= 1e-9:
             # Should not happen if A is properly set, but protects against division by zero
             return np.ones_like(f_x)
             
        # 1. Calculate the argument inside the tan: arg = (pi/2) * f(x)
        arg = (np.pi / 2.0) * f_x
        
        # 2. Apply tan, then scale by 1/A, and shift center back to 1.0
        x = 1.0 + (1.0 / A) * np.tan(arg)
        
        return x.astype(np.float32)

    # Utility method for column indices
    def _get_indices_to_transform(self, group_key: str, all_group_columns: List[str]) -> List[int]:
        """Utility to get the column indices that need transformation."""
        columns_to_transform = self.config[group_key]
        col_to_idx = {col_name: idx for idx, col_name in enumerate(all_group_columns)}
        
        indices_to_transform = [
            col_to_idx[col] for col in columns_to_transform if col in col_to_idx
        ]
        return indices_to_transform

    # ----------------------------------------------------------------------
    # --- PIPELINE HOOKS ---
    # ----------------------------------------------------------------------

    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the forward transformation to X and y and stores A in extra_info.
        """
        samples: SequenceCollection = state.get('samples')
        if not samples:
            print("ArctanMapperAddOn: No 'samples' found in state. Skipping.")
            return state

        # --- Store A in extra_info for inference/evaluation hooks ---
        pipeline_extra_info[A_KEY] = self.A

        feature_col_maps = pipeline_extra_info.get('feature_columns', {})
        transform_y = self.config.get('y', False)

        # --- Iterate over every sample and apply transformations ---
        for sample in samples:
            
            # 1. Transform y (the labels)
            if transform_y and sample.y is not None:
                sample.y = self._transform_func(sample.y)

            # 2. Transform X (the features)
            for group_key in self._target_feature_groups:
                if group_key not in sample.X:
                    continue
                
                data_array = sample.X[group_key]
                all_group_columns = feature_col_maps.get(group_key, [])

                try:
                    indices = self._get_indices_to_transform(group_key, all_group_columns)
                    if not indices:
                        continue
                        
                    # Apply Transformation
                    data_array[:, indices] = self._transform_func(data_array[:, indices])
                    
                except Exception as e:
                    print(f"ArctanMapperAddOn: Error during X transformation of group '{group_key}': {e}")
        
        return state

    def on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the inverse transformation to the model predictions (all_preds_reg)
        and the true labels (all_labels_reg) before metric calculation.
        """
        # Retrieve the scaling factor A saved during the main pipeline transformation
        A = pipeline_extra_info.get(A_KEY)

        if A is None or not self.config.get('y', False):
            return eval_data
            
        print(f"ArctanMapperAddOn: Inverse-transforming labels/predictions with A={A:.4f}...")
        
        # Inverse transform PREDICTIONS
        f_x_preds = eval_data.get("all_preds_reg")
        if f_x_preds is not None and f_x_preds.size > 0:
            eval_data["all_preds_reg"] = self._inverse_transform(f_x_preds, A)
            
        # Inverse transform TRUE LABELS
        f_x_labels = eval_data.get("all_labels_reg")
        if f_x_labels is not None and f_x_labels.size > 0:
            eval_data["all_labels_reg"] = self._inverse_transform(f_x_labels, A)
            
        return eval_data

    def on_server_inference(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the inverse transformation to the model predictions (y_pred_np) 
        before any final scaling.
        """
        # Retrieve the scaling factor A
        A = pipeline_extra_info.get(A_KEY)

        if A is None or not self.config.get('y', False):
            return state
            
        # Assuming the prediction array to be transformed is available here:
        y_pred_np = state.get("y_pred_np") 
        if y_pred_np is None:
             print("ArctanMapperAddOn: No 'y_pred_np' found in state for inverse transformation.")
             return state
             
        print(f"ArctanMapperAddOn: Inverse-transforming y_pred_np with A={A:.4f}...")
        
        # Apply the inverse transform and update the state
        # The result is now in the original ratio scale (e.g., 1.01, 0.99)
        state["y_pred_np"] = self._inverse_transform(y_pred_np, A)
            
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        No-op. Input transformation for features happens in the `transformation` method.
        """
        return state