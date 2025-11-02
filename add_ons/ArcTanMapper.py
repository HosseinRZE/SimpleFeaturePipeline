import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Assuming these imports exist in your environment
from add_ons.base_addon import BaseAddOn 
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample

# --- Constants for Key Names in pipeline_extra_info ---
A_KEY = 'arctan_A_factor' 
B_KEY = 'arctan_B_center' # New key for the center point b

class ArctanMapperAddOn(BaseAddOn):
    """
    Applies the transformation f(x) = (2/pi) * arctan(A * (x - b)).
    The center point 'b' is now configurable.
    
    This function expands values near x=b (controlled by A) and 
    clamps the output strictly between -1 and 1.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, y: bool = False, target_features: Dict[str, List[str]] = {}):
        """
        Initializes the AddOn with explicit parameters and a feature map.
        
        Args:
            a (float): The tunable scaling factor 'A'. Defaults to 1.0.
            b (float): The configurable center point 'b'. Defaults to 1.0.
            y (bool): Whether to transform the target variable 'y'. Defaults to False.
            **feature_config (List[str]): Configuration mapping feature groups to column lists.
                Example: ArctanMapperAddOn(a=10, y=True, main=['col1', 'col2'], aux=['col3'])
        """
        super().__init__() # Added super() call
        self.A = a
        self.b = b
        self.y = y # Use the explicit 'y' argument
        self.on_evaluation_priority = 1
        
        # 'feature_config' (from **kwargs) now *only* contains the feature mapping
        # e.g., {'main': ['col1', 'col2'], 'aux': ['col3']}
        self.target_features = target_features
        
        # _target_feature_groups is just the set of keys from this config
        self._target_feature_groups = list(self.target_features.keys())

    # ----------------------------------------------------------------------
    # --- CORE TRANSFORMATION FUNCTIONS ---
    # ----------------------------------------------------------------------

    def _transform_func(self, x: np.ndarray, A: float = None, b: float = None) -> np.ndarray:
        """
        Applies f(x) = (2/pi) * arctan(A * (x - b)).
        Uses instance A and b if not provided.
        """
        if A is None:
            A = self.A
        if b is None:
            b = self.b
            
        x_float = x.astype(float)
        # (2 / np.pi) is the normalization factor to map the output to (-1, 1)
        return (2.0 / np.pi) * np.arctan(A * (x_float - b))

    def _inverse_transform(self, f_x: np.ndarray, A: float, b: float) -> np.ndarray:
        """
        Calculates the inverse: x = b + (1/A) * tan((pi/2) * f(x)).
        """
        if A is None or A <= 1e-9:
             # If A is too small, arctan was ~0, so x was near b.
             return np.full_like(f_x, b)
             
        # 1. Calculate the argument inside the tan: arg = (pi/2) * f(x)
        arg = (np.pi / 2.0) * f_x
        
        # 2. Apply tan, scale by 1/A, and shift center back to b
        x = b + (1.0 / A) * np.tan(arg)
        
        return x.astype(np.float32)

    # --- Utility Methods (from template) ---

    def _get_indices_to_transform(self, group_key: str, all_group_columns: List[str]) -> List[int]:
        """Utility to get the column indices that need transformation."""
        # FIX: Use self.target_features[group_key] which is now the list of columns.
        # We know group_key is in self.target_features because _target_feature_groups
        # is derived from self.target_features.keys()
        if group_key not in self.target_features:
            return [] # Should not happen if called from main loop, but safe
            
        columns_to_transform = self.target_features[group_key] 
        col_to_idx = {col_name: idx for idx, col_name in enumerate(all_group_columns)}
        
        indices_to_transform = [
            col_to_idx[col] for col in columns_to_transform if col in col_to_idx
        ]
        return indices_to_transform

    def _get_group_columns(self, group_key: str, sample: SequenceSample, pipeline_extra_info: Dict[str, Any]) -> List[str]:
        """Utility to get column names for a group."""
        data = sample.X.get(group_key)
        if isinstance(data, pd.DataFrame):
            return data.columns.tolist()
        if isinstance(data, pd.Series):
            return [data.name] if data.name is not None else []
        # Fallback for numpy arrays
        return pipeline_extra_info.get('feature_columns', {}).get(group_key, [])

    # ----------------------------------------------------------------------
    # --- PIPELINE HOOKS ---
    # ----------------------------------------------------------------------

    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the forward transformation to X and y during training
        and stores A and B in extra_info.
        """
        samples: SequenceCollection = state.get('samples')
        if not samples:
            self.add_trace_print(pipeline_extra_info, "Skipped. No samples found in state.")
            return state
            
        self.add_trace_print(pipeline_extra_info, f"Applying Arctan transform. A={self.A:.3f}, B={self.b:.3f}")

        # --- Store A and B in extra_info for inference/evaluation ---
        pipeline_extra_info[A_KEY] = self.A
        pipeline_extra_info[B_KEY] = self.b
        self.add_trace_print(pipeline_extra_info, f"*** SAVED PARAMS: A={self.A:.6f}, B={self.b:.6f} ***")

        # FIX: Use self.y directly
        transform_y = self.y 

        # --- Iterate over every sample and apply transformations ---
        for i, sample in enumerate(samples):
            
            # 1. Transform y (the labels)
            if transform_y and sample.y is not None:
                sample.y = self._transform_func(sample.y) # Uses self.A, self.b

            # 2. Transform X (the features)
            if not self._target_feature_groups:
                continue
                
            for group_key in self._target_feature_groups:
                if group_key in sample.X:
                    data_object = sample.X[group_key]
                    all_group_columns = self._get_group_columns(group_key, sample, pipeline_extra_info)
                    
                    try:
                        indices = self._get_indices_to_transform(group_key, all_group_columns)
                        if indices:
                            # --- Robust data handling from template ---
                            if isinstance(data_object, pd.DataFrame):
                                cols_to_transform = [all_group_columns[j] for j in indices]
                                data_array = data_object[cols_to_transform].values
                            elif isinstance(data_object, pd.Series):
                                data_array = data_object.values.reshape(-1, 1)
                            else: # Assume numpy array
                                data_array = data_object[:, indices]
                            
                            # Apply Transformation (uses self.A, self.b)
                            transformed_array = self._transform_func(data_array)
                            
                            # --- Assign data back ---
                            if isinstance(data_object, pd.DataFrame):
                                data_object[cols_to_transform] = transformed_array
                            elif isinstance(data_object, pd.Series):
                                sample.X[group_key] = pd.Series(
                                    transformed_array.flatten(), 
                                    index=data_object.index, 
                                    name=data_object.name
                                )
                            else:
                                data_object[:, indices] = transformed_array
                                
                    except Exception as e:
                        self.add_trace_print(pipeline_extra_info, f"🔥TRANSFORM ERROR: Group '{group_key}': {e}")
        
        self.add_trace_print(pipeline_extra_info, f"Successfully applied Arctan transform to {len(samples)} samples.")
        return state

    def on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the inverse transformation to predictions and labels.
        """
        # Retrieve the A and B factors saved during training
        A = pipeline_extra_info.get(A_KEY)
        B = pipeline_extra_info.get(B_KEY) # Retrieve B

        # FIX: Use self.y directly
        if A is None or B is None or not self.y: 
            print("ArctanMapperAddOn: Skipping inverse transform in on_evaluation (A, B not found or y not configured).")
            return eval_data
            
        # Set instance params for consistency (matches template)
        self.A = A
        self.b = B
        
        print(f"ArctanMapperAddOn: Inverse-transforming labels/predictions with A={A:.4f}, B={B:.4f}...")
        
        # Inverse transform PREDICTIONS
        f_x_preds = eval_data.get("all_preds_reg")
        if f_x_preds is not None and f_x_preds.size > 0:
            eval_data["all_preds_reg"] = self._inverse_transform(f_x_preds, A, B)
            
        # Inverse transform TRUE LABELS
        f_x_labels = eval_data.get("all_labels_reg")
        if f_x_labels is not None and f_x_labels.size > 0:
            eval_data["all_labels_reg"] = self._inverse_transform(f_x_labels, A, B)
            
        return eval_data

    def on_server_inference(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the inverse transformation to the model predictions (y_pred_np).
        """
        # Retrieve the scaling factor A and B
        A = pipeline_extra_info.get(A_KEY)
        B = pipeline_extra_info.get(B_KEY) # Retrieve B

        # FIX: Use self.y directly
        if A is None or B is None or not self.y: 
            print("ArctanMapperAddOn: Skipping inverse transform in on_server_inference (A, B not found or y not configured).")
            return state
            
        self.A = A
        self.b = B

        y_pred_np = state.get("y_pred_np") 
        if y_pred_np is None:
             print("ArctanMapperAddOn: No 'y_pred_np' found in state for inverse transformation.")
             return state
                 
        print(f"ArctanMapperAddOn: Inverse-transforming y_pred_np with A={A:.4f}, B={B:.4f}...")
        
        # Apply the inverse transform and update the state
        state["y_pred_np"] = self._inverse_transform(y_pred_np, A, B)
            
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the FORWARD transformation to incoming features
        using the A and B retrieved from the trained pipeline.
        """
        A = pipeline_extra_info.get(A_KEY)
        B = pipeline_extra_info.get(B_KEY) # Retrieve B
        
        # Set instance params from trained pipeline
        self.A = A
        self.b = B

        if A is None or B is None:
            print("ArctanMapperAddOn: Skipping feature transformation in on_server_request (A/B not found).")
            return state
        
        # Check if there are any features to transform
        if not self._target_feature_groups:
             print("ArctanMapperAddOn: No target feature groups configured. Skipping transform in on_server_request.")
             return state
        
        samples: SequenceCollection = state.get('samples')
        if not samples:
            print("ArctanMapperAddOn: No 'samples' found in state for feature transformation.")
            return state
            
        print(f"ArctanMapperAddOn: Applying forward transform to {len(samples)} input features (A={A:.4f}, B={B:.4f})...")
        
        # Use the pipeline's A and B (the ones calculated during training)
        A_to_use = A
        b_to_use = B
        
        for i, sample in enumerate(samples):
            for group_key in self._target_feature_groups:
                if group_key in sample.X:
                    data_object = sample.X[group_key]
                    all_group_columns = self._get_group_columns(group_key, sample, pipeline_extra_info)
                    try:
                        indices = self._get_indices_to_transform(group_key, all_group_columns)
                        if indices:
                            # --- Robust data handling ---
                            if isinstance(data_object, pd.DataFrame):
                                cols_to_transform = [all_group_columns[j] for j in indices]
                                data_array = data_object[cols_to_transform].values
                            elif isinstance(data_object, pd.Series):
                                data_array = data_object.values.reshape(-1, 1)
                            else:
                                data_array = data_object[:, indices]
                            
                            # Use A_to_use and b_to_use in the transformation
                            transformed_array = self._transform_func(data_array, A=A_to_use, b=b_to_use)
                            
                            # --- Assign data back ---
                            if isinstance(data_object, pd.DataFrame):
                                data_object[cols_to_transform] = transformed_array
                            elif isinstance(data_object, pd.Series):
                                sample.X[group_key] = pd.Series(
                                    transformed_array.flatten(), 
                                    index=data_object.index, 
                                    name=data_object.name
                                )
                            else:
                                data_object[:, indices] = transformed_array
                                
                    except Exception as e:
                        print(f"🔥 SERVER FEATURE TRANSFORMATION ERROR: Failed to process group '{group_key}'. Error: {e}")
        return state