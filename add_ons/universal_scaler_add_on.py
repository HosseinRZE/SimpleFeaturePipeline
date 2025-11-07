import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, Normalizer, Binarizer
)
from sklearn.base import BaseEstimator

from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample
from utils.decorators.priority import priority

# ------------------------------------------------------------
# Registry and Keys
# ------------------------------------------------------------
_SUPPORTED_METHODS = {
    "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler,
    "maxabs": MaxAbsScaler, "quantile": QuantileTransformer, "power": PowerTransformer,
    "normalize": Normalizer, "binarize": Binarizer,
}
_INVERTIBLE_METHODS = {
    "standard", "minmax", "robust", "maxabs", "quantile", "power"
}

# --- Key strategy to support both modes ---
# If same_bucket=True:
_MASTER_SCALER_KEY = "master_scaler_state" 
# If same_bucket=False:
_X_SCALER_DICT_KEY = "x_scaler_dict" # A dict of {(group, col): scaler}
_Y_SCALER_KEY = "y_scaler_state"    # A single scaler for y


class ScalerMapperAddOn(BaseAddOn):
    """
    General-purpose AddOn to scale and inverse-scale feature and target data.

    Implements `same_bucket` logic:
    - same_bucket=True (Default): Fits ONE master scaler on ALL specified 
      features AND y values. This one scaler transforms everything.
    - same_bucket=False: Fits an independent scaler for EVERY single column
      and a separate one for y.
    """

    def __init__(self, 
                 method: str = "standard", 
                 features: Dict[str, List[str]] = None, 
                 y: bool = False, 
                 same_bucket: bool = True):
        super().__init__()
        self.method = method.lower()
        self.y = y
        self.features = features or {}
        self.same_bucket = same_bucket
        self._target_feature_groups = list(self.features.keys())
        self.on_server_inference_priority = 2
        self.on_evaluation_priority = 2
        # Validate method
        ScalerClass = _SUPPORTED_METHODS.get(self.method)
        if ScalerClass is None:
            raise ValueError(f"Unknown scaler method '{self.method}'. Supported: {list(_SUPPORTED_METHODS.keys())}")
        
        if self.y and self.method not in _INVERTIBLE_METHODS:
            raise ValueError(
                f"Scaler '{self.method}' cannot be used for y=True as it lacks "
                f"inverse_transform. Use one of: {_INVERTIBLE_METHODS}"
            )

        # --- Instantiate Scaler(s) based on bucket strategy ---
        if self.same_bucket:
            self.master_scaler = ScalerClass()
            self.per_column_scalers = None
            self.y_scaler = None
        else:
            self.master_scaler = None
            self.per_column_scalers: Dict[Tuple[str, str], BaseEstimator] = {}
            for group_key, cols in self.features.items():
                for col_name in cols:
                    self.per_column_scalers[(group_key, col_name)] = ScalerClass()
            self.y_scaler = ScalerClass() if self.y else None
            
    # ------------------------------------------------------------------
    # Utility functions (from pattern)
    # ------------------------------------------------------------------
    
    def _get_group_columns(self, group_key: str, sample: SequenceSample, pipeline_extra_info: Dict[str, Any]) -> List[str]:
        data = sample.X.get(group_key)
        if isinstance(data, pd.DataFrame):
            return data.columns.tolist()
        if isinstance(data, pd.Series):
            return [data.name] if data.name is not None else []
        return pipeline_extra_info.get("feature_columns", {}).get(group_key, [])

    def _get_index_for_column(self, col_name: str, all_group_columns: List[str]) -> int:
        if col_name in all_group_columns:
            return all_group_columns.index(col_name)
        return -1

    # ------------------------------------------------------------------
    # Core transformation (branching logic)
    # ------------------------------------------------------------------
    
    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        
        samples: SequenceCollection = state.get("samples")
        print("sample.y universal",samples.get_by_original_index(1).y)
        if not samples:
            self.add_trace_print(pipeline_extra_info, "No samples found; skipping scaling.")
            return state

        self.add_trace_print(pipeline_extra_info, f"Applying {self.method} scaler (Y={self.y}, same_bucket={self.same_bucket}).")

        if self.same_bucket:
            self._fit_transform_master(samples, pipeline_extra_info)
        else:
            self._fit_transform_per_column(samples, pipeline_extra_info)

        return state

    # --- STRATEGY 1: same_bucket=True ---

    def _fit_transform_master(self, samples: SequenceCollection, pipeline_extra_info: Dict[str, Any]):
        """Fits one scaler to ALL data, then transforms ALL data."""
        
        # 1. FIT
        all_data_to_fit = []
        for sample in samples:
            # Collect X features
            for group_key, cols in self.features.items():
                if group_key in sample.X:
                    data_obj = sample.X[group_key]
                    all_cols = self._get_group_columns(group_key, sample, pipeline_extra_info)
                    for col_name in cols:
                        col_idx = self._get_index_for_column(col_name, all_cols)
                        if col_idx == -1: continue
                        
                        try:
                            if isinstance(data_obj, pd.DataFrame):
                                all_data_to_fit.append(data_obj[[col_name]].values)
                            elif isinstance(data_obj, pd.Series): # Assumes series IS the col
                                all_data_to_fit.append(data_obj.values.reshape(-1, 1))
                            elif isinstance(data_obj, np.ndarray):
                                all_data_to_fit.append(data_obj[:, [col_idx]])
                        except Exception as e:
                            self.add_trace_print(pipeline_extra_info, f"🔥 MasterScaler: Error extracting {group_key}-{col_name}: {e}")
            
            # Collect Y
            if self.y and sample.y is not None and sample.y.size > 0:
                all_data_to_fit.append(sample.y.reshape(-1, 1))

        if not all_data_to_fit:
            self.add_trace_print(pipeline_extra_info, "MasterScaler: No data found to fit.")
            return
            
        try:
            X_train = np.vstack(all_data_to_fit)
            self.master_scaler.fit(X_train)
            self.add_trace_print(pipeline_extra_info, f"Fitted Master scaler on shape {X_train.shape}")
        except Exception as e:
             self.add_trace_print(pipeline_extra_info, f"🔥 MasterScaler: Error fitting: {e}")
             return

        # 2. TRANSFORM
        for sample in samples:
            # Transform X
            for group_key, cols in self.features.items():
                if group_key in sample.X:
                    self._transform_sample_group_master(sample.X[group_key], group_key, cols, sample, pipeline_extra_info)
            # Transform Y
            if self.y and sample.y is not None and sample.y.size > 0:
                sample.y = self.master_scaler.transform(sample.y.reshape(-1, 1)).flatten()
        pipeline_extra_info[_MASTER_SCALER_KEY] = self.master_scaler

    def _transform_sample_group_master(self, data_object: Any, group_key: str, cols: List[str], sample: SequenceSample, pipeline_extra_info: Dict[str, Any]):
        """Helper for same_bucket=True transform"""
        all_cols = self._get_group_columns(group_key, sample, pipeline_extra_info)
        
        for col_name in cols:
            col_idx = self._get_index_for_column(col_name, all_cols)
            if col_idx == -1: continue
            
            try:
                if isinstance(data_object, pd.DataFrame):
                    data_object[col_name] = self.master_scaler.transform(data_object[[col_name]].values)
                elif isinstance(data_object, pd.Series):
                    transformed = self.master_scaler.transform(data_object.values.reshape(-1, 1)).flatten()
                    sample.X[group_key] = pd.Series(transformed, index=data_object.index, name=data_object.name)
                elif isinstance(data_object, np.ndarray):
                    data_object[:, [col_idx]] = self.master_scaler.transform(data_object[:, [col_idx]])
            except Exception as e:
                self.add_trace_print(pipeline_extra_info, f"🔥 MasterScaler: Error transforming {group_key}-{col_name}: {e}")

    # --- STRATEGY 2: same_bucket=False ---

    def _fit_transform_per_column(self, samples: SequenceCollection, pipeline_extra_info: Dict[str, Any]):
        """Fits an independent scaler for each column."""
        
        # 1. FIT X SCALERS
        for (group_key, col_name), scaler in self.per_column_scalers.items():
            data_to_fit = []
            for sample in samples:
                if group_key not in sample.X: continue
                data_obj = sample.X[group_key]
                all_cols = self._get_group_columns(group_key, sample, pipeline_extra_info)
                col_idx = self._get_index_for_column(col_name, all_cols)
                if col_idx == -1: continue
                
                try:
                    if isinstance(data_obj, pd.DataFrame):
                        data_to_fit.append(data_obj[[col_name]].values)
                    elif isinstance(data_obj, pd.Series):
                        data_to_fit.append(data_obj.values.reshape(-1, 1))
                    elif isinstance(data_obj, np.ndarray):
                        data_to_fit.append(data_obj[:, [col_idx]])
                except Exception as e:
                    self.add_trace_print(pipeline_extra_info, f"🔥 PerColScaler: Error extracting {group_key}-{col_name}: {e}")
            
            if data_to_fit:
                try:
                    scaler.fit(np.vstack(data_to_fit))
                except Exception as e:
                    self.add_trace_print(pipeline_extra_info, f"🔥 PerColScaler: Error fitting {group_key}-{col_name}: {e}")

        # 2. FIT Y SCALER
        if self.y:
            y_concat = [s.y.reshape(-1, 1) for s in samples if s.y is not None and s.y.size > 0]
            if y_concat:
                self.y_scaler.fit(np.vstack(y_concat))

        # 3. TRANSFORM
        for sample in samples:
            # Transform X
            for group_key, cols in self.features.items():
                if group_key in sample.X:
                    self._transform_sample_group_per_column(sample.X[group_key], group_key, cols, sample, pipeline_extra_info)
            # Transform Y
            if self.y and sample.y is not None and sample.y.size > 0:
                sample.y = self.y_scaler.transform(sample.y.reshape(-1, 1)).flatten()

        pipeline_extra_info[_X_SCALER_DICT_KEY] = self.per_column_scalers
        if self.y:
            pipeline_extra_info[_Y_SCALER_KEY] = self.y_scaler

    def _transform_sample_group_per_column(self, data_object: Any, group_key: str, cols: List[str], sample: SequenceSample, pipeline_extra_info: Dict[str, Any]):
        """Helper for same_bucket=False transform"""
        all_cols = self._get_group_columns(group_key, sample, pipeline_extra_info)
        
        for col_name in cols:
            scaler = self.per_column_scalers.get((group_key, col_name))
            if scaler is None or not hasattr(scaler, 'n_features_in_'): continue
            
            col_idx = self._get_index_for_column(col_name, all_cols)
            if col_idx == -1: continue
            
            try:
                if isinstance(data_object, pd.DataFrame):
                    data_object[col_name] = scaler.transform(data_object[[col_name]].values)
                elif isinstance(data_object, pd.Series):
                    transformed = scaler.transform(data_object.values.reshape(-1, 1)).flatten()
                    sample.X[group_key] = pd.Series(transformed, index=data_object.index, name=data_object.name)
                elif isinstance(data_object, np.ndarray):
                    data_object[:, [col_idx]] = scaler.transform(data_object[:, [col_idx]])
            except Exception as e:
                self.add_trace_print(pipeline_extra_info, f"🔥 PerColScaler: Error transforming {group_key}-{col_name}: {e}")

    # ------------------------------------------------------------------
    # Inverse & Server Hooks (now check for Master or Y scaler)
    # ------------------------------------------------------------------
    @priority(2) 
    def on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        if not self.y: return eval_data
        
        # Determine which scaler to use
        scaler: Any = pipeline_extra_info.get(_MASTER_SCALER_KEY) # Pri 1: Master
        if scaler is None:
            scaler = pipeline_extra_info.get(_Y_SCALER_KEY) # Pri 2: Y-only
        if scaler is None:
            print("ScalerMapperAddOn: No y_scaler or master_scaler found; skipping inverse transform.")
            return eval_data
        print(f"ScalerMapperAddOn: Inverse-transforming eval data with {self.method}...")
        for key in ["all_preds_reg", "all_labels_reg"]:
            arr = eval_data.get(key)
            if arr is not None and arr.size > 0:
                try:
                    eval_data[key] = scaler.inverse_transform(arr.reshape(-1, 1)).flatten()
                except Exception as e:
                    print(f"ScalerMapperAddOn: Failed inverse-transform on {key}: {e}")
        return eval_data
    
    @priority(2) 
    def on_server_inference(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        if not self.y: return state
        # Determine which scaler to use
        scaler: Any = pipeline_extra_info.get(_MASTER_SCALER_KEY)
        if scaler is None:
            scaler = pipeline_extra_info.get(_Y_SCALER_KEY)
        
        y_pred_np = state.get("y_pred_np")
        if scaler is None or y_pred_np is None:
            return state

        try:
            state["y_pred_np"] = scaler.inverse_transform(y_pred_np.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"ScalerMapperAddOn: Failed inverse-transform on y_pred_np: {e}")
        
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        samples: SequenceCollection = state.get("samples")
        if not samples: return state

        # Check which mode was used during training
        master_scaler = pipeline_extra_info.get(_MASTER_SCALER_KEY)
        
        if master_scaler:
            # --- Restore and run MASTER mode ---
            self.master_scaler = master_scaler
            print(f"ScalerMapperAddOn: Applying forward {self.method} transform (MASTER BUCKET) to server features...")
            for sample in samples:
                for group_key, cols in self.features.items():
                    if group_key in sample.X:
                        self._transform_sample_group_master(sample.X[group_key], group_key, cols, sample, pipeline_extra_info)
        else:
            # --- Restore and run PER-COLUMN mode ---
            per_column_scalers = pipeline_extra_info.get(_X_SCALER_DICT_KEY)
            if not per_column_scalers:
                print("ScalerMapperAddOn: No scalers found; skipping feature scaling.")
                return state
            
            self.per_column_scalers = per_column_scalers
            print(f"ScalerMapperAddOn: Applying forward {self.method} transform (PER-COLUMN) to server features...")
            for sample in samples:
                for group_key, cols in self.features.items():
                    if group_key in sample.X:
                        self._transform_sample_group_per_column(sample.X[group_key], group_key, cols, sample, pipeline_extra_info)
        
        return state