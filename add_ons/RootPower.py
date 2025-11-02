from typing import Dict, Any, List
import numpy as np
import pandas as pd
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_sample import SequenceSample
from data_structure.sequence_collection import SequenceCollection

M_KEY = 'root_power_max_abs_g_x' 
P_KEY = 'root_power_max_abs_g_x_p' # Changed to differentiate from M_KEY
B_KEY = 'root_power_center_b' # New key for the center point b

class RootPowerMapperAddOn(BaseAddOn):
    """
    Applies the two-pass transformation: g(x) = sign(x-b)*|x-b|**p, then f(x)=g(x)/M.
    The center point 'b' is now configurable.
    """
    
    def __init__(self, **kwargs):
        # 2. ADD super().__init__()
        self.config = kwargs
        self.p = self.config.pop('p', 1.0/3.0)
        # New parameter 'b', defaults to 1.0 for backward compatibility
        self.b = self.config.pop('b', 1.0) 
        self.on_evaluation_priority = 1
        if not (0 < self.p < 1):
            raise ValueError("The power 'p' must be between 0 and 1 (e.g., 0.333) to expand variance near x=b.")
        
        self.max_abs_g_x = None 
        
        self._target_feature_groups = {
            key for key, val in self.config.items() if key != 'y' and key != 'b' and key != 'p' and val is not None
        }

    # --- Core Transform Methods (Updated to use self.b) ---
    
    def _transform_numerator(self, x: np.ndarray, b: float = None) -> np.ndarray:
        """g(x) = sign(x-b) * |x-b|**p"""
        if b is None:
            b = self.b
        x_float = x.astype(float)
        delta = x_float - b # Change: use b instead of 1.0
        return np.sign(delta) * np.power(np.abs(delta), self.p)

    def _inverse_transform(self, f_x: np.ndarray, M: float, b: float) -> np.ndarray:
        """Inverse: x = b + sign(g(x)) * |g(x)|**(1/p)"""
        if M is None or M <= 1e-9:
            # If M is too small, it implies g(x) was near zero, so x was near b.
            return np.full_like(f_x, b) 
        g_x = f_x * M
        abs_delta_power_p = np.abs(g_x)
        abs_delta = np.power(abs_delta_power_p, (1.0 / self.p))
        x = b + np.sign(g_x) * abs_delta # Change: use b instead of 1.0
        return x.astype(np.float32)
    
    def _get_indices_to_transform(self, group_key: str, all_group_columns: List[str]) -> List[int]:
        columns_to_transform = self.config[group_key]
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

    # ----------------------------------------------------------------------
    # --- PIPELINE HOOKS ---
    # ----------------------------------------------------------------------
    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
            
        # --- PASS 1 (M calculation) ---
        samples: SequenceCollection = state.get('samples')
        if not samples: self.add_trace_print(pipeline_extra_info, "Skipped. No samples found in state."); return state
        self.add_trace_print(pipeline_extra_info, f"Starting Pass 1 (M calculation) on {len(samples)} samples. P={self.p:.3f}, B={self.b:.3f}")
        transform_y = self.config.get('y', False); max_val = 0.0
        for i, sample in enumerate(samples):
            # Pass the current self.b to _transform_numerator
            if transform_y and sample.y is not None: max_val = max(max_val, np.max(np.abs(self._transform_numerator(sample.y))))
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
                            else: data_array = data_object[:, indices]
                            # Pass the current self.b to _transform_numerator
                            max_val = max(max_val, np.max(np.abs(self._transform_numerator(data_array))))
                    except Exception as e: self.add_trace_print(pipeline_extra_info, f"🔥 PASS 1 ERROR: {e}")
        
        # Store M, P, and B in pipeline_extra_info for later use in inverse/server transforms
        if max_val < 1e-9: 
            M = 1.0
            self.add_trace_print(pipeline_extra_info, f"All input values were {self.b:.6f}. Transformation skipped.");
            pipeline_extra_info[M_KEY] = M
        else: 
            M = max_val
            self.max_abs_g_x = M
            self.add_trace_print(pipeline_extra_info, f"Pass 1 successful. Final Normalization Factor M={M:.6f}")
        
        # Store M, B, and P using their unique keys
        pipeline_extra_info[M_KEY] = M
        pipeline_extra_info[B_KEY] = self.b
        pipeline_extra_info[P_KEY] = self.p
        self.add_trace_print(pipeline_extra_info, f"*** SAVED PARAMS: M={M:.6f}, B={self.b:.6f}, P={self.p:.6f} ***")
        if M < 1e-9: return state

        # --- PASS 2 (Transformation) ---
        logged_first_sample = False 
        TARGET_GROUP = 'main'
        TARGET_COL = 'close_prop'

        for i, sample in enumerate(samples):
            
            # --- Logging Logic (Updated for b) ---
            if not logged_first_sample and i == 0:
                group_key_to_log = TARGET_GROUP
                if group_key_to_log in sample.X and TARGET_COL in self.config.get(TARGET_GROUP, []):
                    try:
                        all_group_columns = self._get_group_columns(group_key_to_log, sample, pipeline_extra_info)
                        data_object = sample.X[group_key_to_log]
                        log_col_index = all_group_columns.index(TARGET_COL)
                        original_value = data_object.iloc[0][TARGET_COL]
                        # Use self.b in the trace logging
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
            # Pass the current self.b to _transform_numerator (implicitly used as default)
            if transform_y and sample.y is not None: sample.y = self._transform_numerator(sample.y) / M
            for group_key in self._target_feature_groups:
                if group_key in sample.X:
                    data_object = sample.X[group_key]
                    all_group_columns = self._get_group_columns(group_key, sample, pipeline_extra_info)
                    try:
                        indices = self._get_indices_to_transform(group_key, all_group_columns)
                        if indices:
                            if isinstance(data_object, (pd.DataFrame, pd.Series)): cols_to_transform = [all_group_columns[i] for i in indices]; data_array = data_object[cols_to_transform].values
                            elif isinstance(data_object, pd.Series): data_array = data_object.values.reshape(-1, 1)
                            else: data_array = data_object[:, indices]
                            # Pass the current self.b to _transform_numerator (implicitly used as default)
                            transformed_array = self._transform_numerator(data_array) / M
                            if isinstance(data_object, pd.DataFrame): data_object[cols_to_transform] = transformed_array
                            elif isinstance(data_object, pd.Series): sample.X[group_key] = pd.Series(transformed_array.flatten(), index=data_object.index, name=data_object.name)
                            else: data_object[:, indices] = transformed_array
                    except Exception as e: self.add_trace_print(pipeline_extra_info, f"🔥 PASS 2 MUTATION ERROR: {e}")
                        
        if not logged_first_sample:
            self.add_trace_print(pipeline_extra_info, f"Successfully applied transformation to {len(samples)} samples. M={M:.4f}, b={self.b:.4f}")

        return state

    # --- PIPELINE INVERSE HOOKS (Updated to retrieve and use B) ---

    def on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        M = pipeline_extra_info.get(M_KEY)
        B = pipeline_extra_info.get(B_KEY)
        P = pipeline_extra_info.get(P_KEY) 
        # Check P as well, since it's now essential for inverse_transform through self.p
        if M is None or B is None or P is None or not self.config.get('y', False): # <<< MODIFIED: Added P check
            print("RootPowerMapperAddOn: Skipping inverse transform in on_evaluation (M, B, P not found or y not configured).")
            return eval_data
        # Update self.p for consistency
        self.p = P 
        self.b = B
        print(f"RootPowerMapperAddOn: Inverse-transforming labels/predictions with M={M:.4f}, p={self.p:.4f}, b={B:.4f}...")
        f_x_preds = eval_data.get("all_preds_reg")
        if f_x_preds is not None and f_x_preds.size > 0:
            eval_data["all_preds_reg"] = self._inverse_transform(f_x_preds, M, B)
        f_x_labels = eval_data.get("all_labels_reg")
        if f_x_labels is not None and f_x_labels.size > 0:
            eval_data["all_labels_reg"] = self._inverse_transform(f_x_labels, M, B)
        return eval_data

    def on_server_inference(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        M = pipeline_extra_info.get(M_KEY)
        B = pipeline_extra_info.get(B_KEY)
        P = pipeline_extra_info.get(P_KEY)
        self.p = P 
        self.b = B
        if M is None or B is None or P is None or not self.config.get('y', False):
            print("RootPowerMapperAddOn: Skipping inverse transform in on_server_inference (M, B, P not found or y not configured).")
            return state
        y_pred_np = state.get("y_pred_np") 
        if y_pred_np is None:
            print("RootPowerMapperAddOn: No 'y_pred_np' found in state for inverse transformation.")
            return state
        print(f"RootPowerMapperAddOn: Inverse-transforming y_pred_np with M={M:.4f}, p={self.p:.4f}, b={B:.4f}...")
        state["y_pred_np"] = self._inverse_transform(y_pred_np, M, B) # Pass B
        return state

    # --- PIPELINE FORWARD HOOK (Updated to retrieve and use B) ---

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        M = pipeline_extra_info.get(M_KEY)
        P = pipeline_extra_info.get(P_KEY) or self.p
        B = pipeline_extra_info.get(B_KEY) # Retrieve B
        self.p = P
        self.b = B
        if M is None or M < 1e-9 or B is None:
            print("RootPowerMapperAddOn: Skipping feature transformation in on_server_request (M/B not found or M too small).")
            return state
        
        samples: SequenceCollection = state.get('samples')
        if not samples:
            print("RootPowerMapperAddOn: No 'samples' found in state for feature transformation.")
            return state
            
        print(f"RootPowerMapperAddOn: Applying forward transform to {len(samples)} input features (M={M:.4f}, p={P:.4f}, b={B:.4f})...")
        
        # Use the pipeline's B (the one calculated during training)
        b_to_use = B 
        
        for i, sample in enumerate(samples):
            for group_key in self._target_feature_groups:
                if group_key in sample.X:
                    data_object = sample.X[group_key]
                    all_group_columns = self._get_group_columns(group_key, sample, pipeline_extra_info)
                    try:
                        indices = self._get_indices_to_transform(group_key, all_group_columns)
                        if indices:
                            # ... (data extraction logic remains the same) ...
                            if isinstance(data_object, pd.DataFrame):
                                cols_to_transform = [all_group_columns[j] for j in indices]
                                data_array = data_object[cols_to_transform].values
                            elif isinstance(data_object, pd.Series):
                                data_array = data_object.values.reshape(-1, 1)
                            else:
                                data_array = data_object[:, indices]
                            
                            # Use b_to_use in the transformation
                            transformed_array = self._transform_numerator(data_array, b=b_to_use) / M
                            
                            # ... (data assignment logic remains the same) ...
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