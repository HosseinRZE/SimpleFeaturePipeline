from typing import Dict, Any, List
import numpy as np
import pandas as pd
# 1. IMPORT BaseAddOn AND the placeholder
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_sample import SequenceSample
from data_structure.sequence_collection import SequenceCollection

M_KEY = 'root_power_max_abs_g_x' 
P_KEY = 'root_power_max_abs_g_x' 

class RootPowerMapperAddOn(BaseAddOn):
    """
    Applies the two-pass transformation: g(x) = sign(x-1)*|x-1|**p, then f(x)=g(x)/M.
    """
    
    def __init__(self, **kwargs):
        # 2. ADD super().__init__()
        self.config = kwargs
        self.p = self.config.pop('p', 1.0/3.0)
        self.on_evaluation_priority = 1
        if not (0 < self.p < 1):
             raise ValueError("The power 'p' must be between 0 and 1 (e.g., 0.333) to expand variance near x=1.")
        
        self.max_abs_g_x = None 
        
        self._target_feature_groups = {
            key for key, val in self.config.items() if key != 'y' and val is not None
        }

    # ... (core transform, inverse_transform, and utility methods are unchanged) ...
    def _transform_numerator(self, x: np.ndarray) -> np.ndarray:
        x_float = x.astype(float)
        delta = x_float - 1.0
        return np.sign(delta) * np.power(np.abs(delta), self.p)

    def _inverse_transform(self, f_x: np.ndarray, M: float) -> np.ndarray:
        if M is None or M <= 1e-9:
            return np.ones_like(f_x) 
        g_x = f_x * M
        abs_delta_power_p = np.abs(g_x)
        abs_delta = np.power(abs_delta_power_p, (1.0 / self.p))
        x = 1.0 + np.sign(g_x) * abs_delta
        return x.astype(np.float32)

    def _get_group_columns(self, group_key: str, sample: SequenceSample, pipeline_extra_info: Dict[str, Any]) -> List[str]:
        data = sample.X.get(group_key)
        if isinstance(data, pd.DataFrame):
            return data.columns.tolist()
        if isinstance(data, pd.Series):
            return [data.name] if data.name is not None else []
        return pipeline_extra_info.get('feature_columns', {}).get(group_key, [])

    def _get_indices_to_transform(self, group_key: str, all_group_columns: List[str]) -> List[int]:
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
            
            # ... (Pass 1 logic is unchanged) ...
            samples: SequenceCollection = state.get('samples')
            if not samples: self.add_trace_print(pipeline_extra_info, "Skipped. No samples found in state."); return state
            self.add_trace_print(pipeline_extra_info, f"Starting Pass 1 (M calculation) on {len(samples)} samples. P={self.p:.3f}")
            transform_y = self.config.get('y', False); max_val = 0.0
            for i, sample in enumerate(samples):
                if i == 0 and 'main' not in sample.X: self.add_trace_print(pipeline_extra_info, f"⚠️ KEY MISMATCH ALERT: 'main' not in sample.X keys.")
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
                                max_val = max(max_val, np.max(np.abs(self._transform_numerator(data_array))))
                        except Exception as e: self.add_trace_print(pipeline_extra_info, f"🔥 PASS 1 ERROR: {e}")
            if max_val < 1e-9: M = 1.0; self.add_trace_print(pipeline_extra_info, "All input values were 1.0. Transformation skipped."); pipeline_extra_info[M_KEY] = M; pipeline_extra_info[P_KEY] = self.p; return state
            else: M = max_val; self.max_abs_g_x = M; self.add_trace_print(pipeline_extra_info, f"Pass 1 successful. Final Normalization Factor M={M:.6f}"); pipeline_extra_info[M_KEY] = M; pipeline_extra_info[P_KEY] = self.p


        # --- PASS 2 ---
            logged_first_sample = False 
            TARGET_GROUP = 'main'
            TARGET_COL = 'close_prop'

            for i, sample in enumerate(samples):
                
                if not logged_first_sample and i == 0:
                    group_key_to_log = TARGET_GROUP
                    if group_key_to_log in sample.X:
                        try:
                            all_group_columns = self._get_group_columns(group_key_to_log, sample, pipeline_extra_info)
                            data_object = sample.X[group_key_to_log]

                            # --- Reverted Logging Logic ---
                            
                            # Get single value trace string
                            log_col_index = all_group_columns.index(TARGET_COL)
                            original_value = data_object.iloc[0][TARGET_COL]
                            g_x = self._transform_numerator(np.array([original_value]))[0]
                            transformed_value = g_x / M
                            first_value_trace_string = f"'{TARGET_COL}': x={original_value:.6f} -> f(x)={transformed_value:.6f} (M={M:.4f})"
                            
                            # Removed all df_comparison, set_attachment, and combined print logic.
                            
                            # Log *only* the single value trace string.
                            # This is what was appearing in your table.
                            self.add_trace_print(pipeline_extra_info, first_value_trace_string)
                            logged_first_sample = True
                            
                        except Exception as e:
                            self.add_trace_print(pipeline_extra_info, f"🔥 DEBUG ERROR: Failed to log trace. Error: {e}")
                            logged_first_sample = True
                    else:
                        self.add_trace_print(pipeline_extra_info, f"🔥 DEBUG ERROR: Target group '{TARGET_GROUP}' not in sample.")
                
                # ... (Rest of Pass 2 transformation loop) ...
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
                                transformed_array = self._transform_numerator(data_array) / M
                                if isinstance(data_object, pd.DataFrame): data_object[cols_to_transform] = transformed_array
                                elif isinstance(data_object, pd.Series): sample.X[group_key] = pd.Series(transformed_array.flatten(), index=data_object.index, name=data_object.name)
                                else: data_object[:, indices] = transformed_array
                        except Exception as e: self.add_trace_print(pipeline_extra_info, f"🔥 PASS 2 MUTATION ERROR: {e}")
                            
            if not logged_first_sample:
                self.add_trace_print(pipeline_extra_info, f"Successfully applied transformation to {len(samples)} samples. M={M:.4f}")

            return state

    # ... (on_evaluation, on_server_inference, on_server_request methods are unchanged) ...
    def on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        M = pipeline_extra_info.get(M_KEY)
        self.on_evaluation_priority = 1
        if M is None or not self.config.get('y', False):
            print("RootPowerMapperAddOn: Skipping inverse transform in on_evaluation (M not found or y not configured).")
            return eval_data
        print(f"RootPowerMapperAddOn: Inverse-transforming labels/predictions with M={M:.4f}, p={self.p:.4f}...")
        f_x_preds = eval_data.get("all_preds_reg")
        if f_x_preds is not None and f_x_preds.size > 0:
            eval_data["all_preds_reg"] = self._inverse_transform(f_x_preds, M)
        f_x_labels = eval_data.get("all_labels_reg")
        if f_x_labels is not None and f_x_labels.size > 0:
            eval_data["all_labels_reg"] = self._inverse_transform(f_x_labels, M)
        return eval_data

    def on_server_inference(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        M = pipeline_extra_info.get(M_KEY)
        if M is None or not self.config.get('y', False):
            print("RootPowerMapperAddOn: Skipping inverse transform in on_server_inference (M not found or y not configured).")
            return state
        y_pred_np = state.get("y_pred_np") 
        if y_pred_np is None:
             print("RootPowerMapperAddOn: No 'y_pred_np' found in state for inverse transformation.")
             return state
        print(f"RootPowerMapperAddOn: Inverse-transforming y_pred_np with M={M:.4f}, p={self.p:.4f}...")
        state["y_pred_np"] = self._inverse_transform(y_pred_np, M)
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        M = pipeline_extra_info.get(M_KEY)
        p = pipeline_extra_info.get(P_KEY) or self.p
        if M is None or M < 1e-9:
            print("RootPowerMapperAddOn: Skipping feature transformation in on_server_request (M not found or too small).")
            return state
        samples: SequenceCollection = state.get('samples')
        if not samples:
            print("RootPowerMapperAddOn: No 'samples' found in state for feature transformation.")
            return state
        print(f"RootPowerMapperAddOn: Applying forward transform to {len(samples)} input features (M={M:.4f}, p={p:.4f})...")
        for i, sample in enumerate(samples):
            for group_key in self._target_feature_groups:
                if group_key in sample.X:
                    data_object = sample.X[group_key]
                    all_group_columns = self._get_group_columns(group_key, sample, pipeline_extra_info)
                    try:
                        indices = self._get_indices_to_transform(group_key, all_group_columns)
                        if indices:
                            if isinstance(data_object, pd.DataFrame):
                                cols_to_transform = [all_group_columns[j] for j in indices]
                                data_array = data_object[cols_to_transform].values
                            elif isinstance(data_object, pd.Series):
                                data_array = data_object.values.reshape(-1, 1)
                            else:
                                data_array = data_object[:, indices]
                            transformed_array = self._transform_numerator(data_array) / M
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