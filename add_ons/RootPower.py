from typing import Dict, Any, List
import numpy as np
import pandas as pd
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_sample import SequenceSample
from data_structure.sequence_collection import SequenceCollection
from utils.decorators.priority import priority

M_KEY = 'root_power_max_abs_g_x' 
P_KEY = 'root_power_max_abs_g_x_p' 
B_KEY = 'root_power_center_b' 

class RootPowerMapperAddOn(BaseAddOn):
    """
    Applies the two-pass transformation: g(x) = sign(x-b)*|x-b|**p, then f(x)=g(x)/M.
    Configuration is passed directly to the constructor.
    """
    def __init__(self, 
                 p: float = 0.33, 
                 b: float = 1, 
                 transform_y: bool = False, 
                 target_features: Dict[str, List[str]] = None):
        super().__init__()
        self.on_server_inference_priority = 1
        self.on_evaluation_priority =1
        self.p = p
        self.b = b
        self.transform_y = transform_y 
        if not (0 < self.p < 1):
            raise ValueError("The power 'p' must be between 0 and 1 (e.g., 0.333) to expand variance near x=b.")
        self.max_abs_g_x = None 
        if target_features is None:
            self.target_features = {}
        else:
            self.target_features = target_features
        self._target_feature_groups = list(self.target_features.keys())
    
    def _transform_numerator(self, x: np.ndarray, b: float = None) -> np.ndarray:
        """g(x) = sign(x-b) * |x-b|**p"""
        # Use the provided 'b' or fall back to the instance's 'b'
        b_to_use = b if b is not None else self.b
        x_float = x.astype(float)
        delta = x_float - b_to_use 
        return np.sign(delta) * np.power(np.abs(delta), self.p)

    def _inverse_transform(self, f_x: np.ndarray, M: float, b: float) -> np.ndarray:
        """Inverse: x = b + sign(g(x)) * |g(x)|**(1/p)"""
        if M is None or M <= 1e-9:
            return np.full_like(f_x, b) 
        g_x = f_x * M
        abs_delta_power_p = np.abs(g_x)
        abs_delta = np.power(abs_delta_power_p, (1.0 / self.p))
        x = b + np.sign(g_x) * abs_delta
        return x.astype(np.float32)
    
    # --- Utility Helpers ---
    
    def _get_indices_to_transform(self, group_key: str, all_group_columns: List[str]) -> List[int]:
        columns_to_transform = self.target_features.get(group_key, []) 
        col_to_idx = {col_name: idx for idx, col_name in enumerate(all_group_columns)}
        indices_to_transform = [
            col_to_idx[col] for col in columns_to_transform if col in col_to_idx
        ]
        return indices_to_transform
    
    def _get_group_columns(self, group_key: str, sample: SequenceSample, pipeline_extra_info: Dict[str, Any]) -> List[str]:
        data = sample.X.get(group_key)
        if isinstance(data, pd.DataFrame):
            return data.columns.tolist()
        if isinstance(data, pd.Series):
            return [data.name] if data.name is not None else []
        return pipeline_extra_info.get('feature_columns', {}).get(group_key, [])

    # ------------------------------------------------------------------
    # --- Core Fit/Transform Logic (Refactored) ---
    # ------------------------------------------------------------------

    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full fit-and-transform process.
        """
        samples: SequenceCollection = state.get('samples')
        
        # --- PASS 1: FIT ---
        # Calculate M, B, P and save them to self and pipeline_extra_info
        fit_successful = self._fit(samples, pipeline_extra_info)
        
        # --- PASS 2: TRANSFORM ---
        # Apply the transformation using the parameters found in _fit
        if fit_successful:
            self._transform(samples, pipeline_extra_info)
        
        return state

    def _fit(self, samples: SequenceCollection, pipeline_extra_info: Dict[str, Any]) -> bool:
        """
        PASS 1: Calculates the normalization factor 'M' by finding the
        max absolute value of g(x) across all target data.
        Returns True if successful, False if skipped.
        """
        if not samples: 
            self.add_trace_print(pipeline_extra_info, "Skipped. No samples found in state.")
            return False

        self.add_trace_print(pipeline_extra_info, f"Starting Pass 1 (M calculation) on {len(samples)} samples. P={self.p:.3f}, B={self.b:.3f}")
        max_val = 0.0
        
        for i, sample in enumerate(samples):
            # Pass the current self.b to _transform_numerator
            if self.transform_y and sample.y is not None: 
                max_val = max(max_val, np.max(np.abs(self._transform_numerator(sample.y))))
            
            for group_key in self._target_feature_groups:
                if group_key in sample.X:
                    all_group_columns = self._get_group_columns(group_key, sample, pipeline_extra_info)
                    try:
                        indices = self._get_indices_to_transform(group_key, all_group_columns)
                        if i == 0 and indices: self.add_trace_print(pipeline_extra_info, f"Pass 1 Debug: Group '{group_key}' has {len(indices)} cols to transform.")
                        if indices:
                            data_object = sample.X[group_key]
                            if isinstance(data_object, (pd.DataFrame, pd.Series)):
                                cols_to_transform = [all_group_columns[idx] for idx in indices]
                                data_array = data_object[cols_to_transform].values if isinstance(data_object, pd.DataFrame) else data_object.values.reshape(-1, 1)
                            else: 
                                data_array = data_object[:, indices]
                            # Pass the current self.b to _transform_numerator
                            max_val = max(max_val, np.max(np.abs(self._transform_numerator(data_array))))
                    except Exception as e: 
                        self.add_trace_print(pipeline_extra_info, f"🔥 PASS 1 ERROR: {e}")
        
        # --- Finalize M and save parameters ---
        if max_val < 1e-9: 
            M = 1.0
            self.add_trace_print(pipeline_extra_info, f"All input values were {self.b:.6f}. Transformation skipped.");
        else: 
            M = max_val
            self.add_trace_print(pipeline_extra_info, f"Pass 1 successful. Final Normalization Factor M={M:.6f}")
            
        self.max_abs_g_x = M
        pipeline_extra_info[M_KEY] = M
        pipeline_extra_info[B_KEY] = self.b
        pipeline_extra_info[P_KEY] = self.p
        self.add_trace_print(pipeline_extra_info, f"*** SAVED PARAMS: M={M:.6f}, B={self.b:.6f}, P={self.p:.6f} ***")
        
        return M >= 1e-9 # Return success status

    def _transform(self, samples: SequenceCollection, pipeline_extra_info: Dict[str, Any]):
        """
        PASS 2: Applies the transformation f(x) = g(x) / M to all target data.
        Relies on self.max_abs_g_x, self.b, and self.p being set.
        """
        M = self.max_abs_g_x
        if M is None:
             self.add_trace_print(pipeline_extra_info, "🔥 PASS 2 ERROR: 'M' is not set. Skipping transform.")
             return
             
        logged_first_sample = False 
        TARGET_GROUP = 'main'
        TARGET_COL = 'close_prop'

        for i, sample in enumerate(samples):
            
            # --- Logging Logic ---
            if not logged_first_sample and i == 0:
                group_key_to_log = TARGET_GROUP
                if group_key_to_log in sample.X and TARGET_COL in self.target_features.get(TARGET_GROUP, []):
                    try:
                        all_group_columns = self._get_group_columns(group_key_to_log, sample, pipeline_extra_info)
                        data_object = sample.X[group_key_to_log]
                        log_col_index = all_group_columns.index(TARGET_COL)
                        original_value = data_object.iloc[0][TARGET_COL]
                        # Use self.b (which is set)
                        g_x = self._transform_numerator(np.array([original_value]))[0]
                        transformed_value = g_x / M
                        first_value_trace_string = f"'{TARGET_COL}': x={original_value:.6f} -> f(x)={transformed_value:.6f} (M={M:.4f}, b={self.b:.4f})"
                        self.add_trace_print(pipeline_extra_info, first_value_trace_string)
                        logged_first_sample = True
                    except Exception as e:
                        self.add_trace_print(pipeline_extra_info, f"🔥 DEBUG ERROR: Failed to log trace. Error: {e}")
                        logged_first_sample = True
                elif not logged_first_sample:
                    self.add_trace_print(pipeline_extra_info, f"🔥 DEBUG: Target column '{TARGET_COL}' not in config for group '{TARGET_GROUP}'. Skipping trace log.")
                    logged_first_sample = True

            # --- Transformation Logic (Pass 2) ---
            # _transform_numerator will use self.b
            if self.transform_y and sample.y is not None: 
                sample.y = self._transform_numerator(sample.y) / M
            
            for group_key in self._target_feature_groups:
                if group_key in sample.X:
                    data_object = sample.X[group_key]
                    all_group_columns = self._get_group_columns(group_key, sample, pipeline_extra_info)
                    try:
                        indices = self._get_indices_to_transform(group_key, all_group_columns)
                        if indices:
                            if isinstance(data_object, pd.DataFrame): 
                                cols_to_transform = [all_group_columns[i] for i in indices]
                                data_array = data_object[cols_to_transform].values
                            elif isinstance(data_object, pd.Series): 
                                data_array = data_object.values.reshape(-1, 1)
                            else: 
                                data_array = data_object[:, indices]
                            
                            transformed_array = self._transform_numerator(data_array) / M
                            
                            if isinstance(data_object, pd.DataFrame): 
                                data_object[cols_to_transform] = transformed_array
                            elif isinstance(data_object, pd.Series): 
                                sample.X[group_key] = pd.Series(transformed_array.flatten(), index=data_object.index, name=data_object.name)
                            else: 
                                data_object[:, indices] = transformed_array
                    except Exception as e: 
                        self.add_trace_print(pipeline_extra_info, f"🔥 PASS 2 MUTATION ERROR: {e}")
                        
        if not logged_first_sample:
            self.add_trace_print(pipeline_extra_info, f"Successfully applied transformation to {len(samples)} samples. M={M:.4f}, b={self.b:.4f}")

    # --- PIPELINE INVERSE HOOKS ---
    @priority(3)
    def on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        M = pipeline_extra_info.get(M_KEY)
        B = pipeline_extra_info.get(B_KEY)
        P = pipeline_extra_info.get(P_KEY) 
        
        if M is None or B is None or P is None or not self.transform_y: 
            print("RootPowerMapperAddOn: Skipping inverse transform in on_evaluation (M, B, P not found or y not configured).")
            return eval_data
            
        self.p = P # Restore p just in case
        # self.b = B (not needed, as B is passed directly)
        
        print(f"RootPowerMapperAddOn: Inverse-transforming labels/predictions with M={M:.4f}, p={self.p:.4f}, b={B:.4f}...")
        f_x_preds = eval_data.get("all_preds_reg")
        if f_x_preds is not None and f_x_preds.size > 0:
            eval_data["all_preds_reg"] = self._inverse_transform(f_x_preds, M, B)
        f_x_labels = eval_data.get("all_labels_reg")
        if f_x_labels is not None and f_x_labels.size > 0:
            eval_data["all_labels_reg"] = self._inverse_transform(f_x_labels, M, B)
        return eval_data
    @priority(3)
    def on_server_inference(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        M = pipeline_extra_info.get(M_KEY)
        B = pipeline_extra_info.get(B_KEY)
        P = pipeline_extra_info.get(P_KEY)
        
        if M is None or B is None or P is None or not self.transform_y:
            print("RootPowerMapperAddOn: Skipping inverse transform in on_server_inference (M, B, P not found or y not configured).")
            return state
            
        self.p = P # Restore p
        
        y_pred_np = state.get("y_pred_np") 
        if y_pred_np is None:
            print("RootPowerMapperAddOn: No 'y_pred_np' found in state for inverse transformation.")
            return state
        print(f"RootPowerMapperAddOn: Inverse-transforming y_pred_np with M={M:.4f}, p={self.p:.4f}, b={B:.4f}...")
        state["y_pred_np"] = self._inverse_transform(y_pred_np, M, B).flatten()
        return state
    
    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the FORWARD transformation to new request data using
        the parameters saved from training.
        """
        M = pipeline_extra_info.get(M_KEY)
        P = pipeline_extra_info.get(P_KEY)
        B = pipeline_extra_info.get(B_KEY)
        
        if M is None or M < 1e-9 or B is None or P is None:
            print("RootPowerMapperAddOn: Skipping feature transformation in on_server_request (M/B/P not found or M too small).")
            return state
        
        # --- Restore saved state from training ---
        self.max_abs_g_x = M
        self.p = P
        self.b = B
        
        samples: SequenceCollection = state.get('samples')
        if not samples:
            print("RootPowerMapperAddOn: No 'samples' found in state for feature transformation.")
            return state
            
        print(f"RootPowerMapperAddOn: Applying forward transform to {len(samples)} input features (M={M:.4f}, p={P:.4f}, b={B:.4f})...")
        
        # --- RE-USE THE TRANSFORM METHOD ---
        self._transform(samples, pipeline_extra_info)
        
        return state