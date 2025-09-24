import pandas as pd 
import importlib
from utils.make_step import make_step
from add_ons.feature_pipeline5 import FeaturePipeline
import numpy as np
import logging
logger = logging.getLogger(__name__)  

class ServerPreprocess:
    def __init__(self, feature_pipeline, scalers_path=None):
        """
        Stateful preprocessing for serving.
        - feature_pipeline: same FeaturePipeline used in training
        - scalers_path: path to saved scalers (from training)
        """
        self.feature_pipeline = feature_pipeline
        self.dataset = pd.DataFrame()           # processed features
        self.reference_dataset = pd.DataFrame() # raw OHLC candles
        self._first_prepare_seq = True 
        # Load pre-fitted scalers from training
        if scalers_path is not None:
            self.feature_pipeline.load(scalers_path)

    def add_candle(self, new_candle: pd.Series | dict):
        """
        Add one new candle to reference_dataset, update processed dataset.
        - Applies only global steps to full reference_dataset via fit().
        - Maintains self.dataset as dict of DataFrames: {"main": ..., **global_dicts}.
        - Prints info info to track pipeline transformations.
        """
        if isinstance(new_candle, dict):
            new_candle_df = pd.DataFrame([new_candle])
        elif isinstance(new_candle, pd.Series):
            new_candle_df = pd.DataFrame([new_candle.to_dict()])
        else:
            raise ValueError("new_candle must be dict or pd.Series")

        # Store raw candle
        self.add_reference_candle(new_candle_df)
        self.feature_pipeline.fit(self.reference_dataset, normalize = False)
        # Update processed dataset dict
        self.dataset = self.feature_pipeline.global_data      

    def add_reference_candle(self, new_candle: pd.DataFrame | dict | pd.Series):
        """
        Store raw candle (without applying feature pipeline).
        """
        if isinstance(new_candle, dict):
            new_candle = pd.DataFrame([new_candle])
        elif isinstance(new_candle, pd.Series):
            new_candle = pd.DataFrame([new_candle.to_dict()])
        elif isinstance(new_candle, pd.DataFrame):
            new_candle = new_candle.copy()
        else:
            raise ValueError("new_candle must be dict, Series, or DataFrame")

        self.reference_dataset = pd.concat([self.reference_dataset, new_candle], ignore_index=True)

    def prepare_seq(self, seq_len: int):
        # --- Check lengths ---
        lengths = {k: len(df) for k, df in self.dataset.items()}
        if any(l < seq_len for l in lengths.values()):
            raise ValueError(f"Not enough data: have {lengths}, need {seq_len}")

        # --- Slice last seq_len rows ---
        seq_slice = {k: v.iloc[-seq_len:].copy() for k, v in self.dataset.items()}

        # --- Apply per-window steps ---
        seq_dict = self.feature_pipeline.apply_window(seq_slice)

        # --- Apply scalers ---
        seq_dict = self.perform_global_scaler(seq_dict)
        seq_dict = self.perform_window_scaler(seq_dict)

        return seq_dict


    def prepare_xgboost_seq(self, seq_len: int, model=None) -> np.ndarray:
        """
        Prepare a sequence for XGBoost prediction.
        - Takes last `seq_len` rows from dataset
        - Applies per-window steps and normalization
        - Applies the exact transformation pipeline (ROCKET, flatten, etc.)
        - Optionally filters to model's expected features
        - Returns flat NumPy vector (shape: [1, n_features])
        """
        lengths = {k: len(df) for k, df in self.dataset.items()}
        if any(l < seq_len for l in lengths.values()):
            raise ValueError(f"Not enough data: have {lengths}, need {seq_len}")

        # Slice last `seq_len` rows
        seq_slice = {k: v.iloc[-seq_len:].copy() for k, v in self.dataset.items()}

        # Apply per-window steps
        seq_dict = self.feature_pipeline.apply_window(seq_slice)

        # Apply scalers
        seq_dict = self.perform_global_scaler(seq_dict)
        seq_dict = self.perform_window_scaler(seq_dict)

        # Wrap into list since transformations expect multiple sequences
        X_trans = self.feature_pipeline.apply_transformations([seq_dict])

        # --- If model is provided, align features ---
        if model is not None:
            if hasattr(model, "feature_names_in_"):  # sklearn API
                expected = list(model.feature_names_in_)
                X_trans = X_trans[expected]
            else:
                # fallback: clip to correct number of features
                n_feats = model.get_booster().num_features()
                X_trans = X_trans.iloc[:, :n_feats]

        return X_trans.to_numpy(dtype=np.float32).reshape(1, -1)
    
    def perform_global_scaler(self, seq_dict: dict) -> dict:
        """
        Apply global scalers to each dict in seq_dict (in-place).
        Includes debug prints only on the first run.
        """
        printed = False
        for k, df in seq_dict.items():
            norm_cfg = self.feature_pipeline.norm_methods.get(k, {})
            if norm_cfg:
                if self._first_prepare_seq:
                    printed = True
                    print(f">>> Global normalization for dict: {k}")
                    for col, method in norm_cfg.items():
                        print(f"   column: {col}, method: {method}")
                        scaler = None
                        if getattr(self.feature_pipeline, "scalers", None):
                            key = f"{k}.{col}"
                            scaler = self.feature_pipeline.scalers.get(key, None)
                        if scaler is not None:
                            if hasattr(scaler, "mean_"):
                                print(f"      -> StandardScaler | mean={scaler.mean_[0]:.6f}, var={scaler.var_[0]:.6f}")
                            elif hasattr(scaler, "min_") and hasattr(scaler, "scale_"):
                                print(f"      -> MinMaxScaler   | min={scaler.min_[0]:.6f}, scale={scaler.scale_[0]:.6f}")
                            elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
                                print(f"      -> RobustScaler   | center={scaler.center_[0]:.6f}, scale={scaler.scale_[0]:.6f}")
                            else:
                                print(f"      -> {type(scaler).__name__} (no standard stats exposed)")

                seq_dict[k] = self.feature_pipeline._normalize_single(
                    df, norm_cfg, fit=False, dict_name=k
                )

        if printed:
            self._first_prepare_seq = False
        return seq_dict

    def perform_window_scaler(self, seq_dict: dict) -> dict:
        """
        Apply window-based scalers to each dict in seq_dict (in-place).
        Includes debug prints only on the first run.
        """
        if not getattr(self.feature_pipeline, "window_scalers", None):
            return seq_dict

        printed = False
        for dict_name, df in seq_dict.items():
            for (dname, col_name), scaler in self.feature_pipeline.window_scalers.items():
                if dname != dict_name or col_name not in df.columns:
                    continue

                before = df[col_name].values[:5].copy()
                df[col_name] = scaler.transform(df[[col_name]].values)
                after = df[col_name].values[:5]

                if self._first_prepare_seq:
                    printed = True
                    print(f"{dict_name} {col_name} before WINDOW norm: {before}")
                    print(f"{dict_name} {col_name} after  WINDOW norm: {after}")
                    print("window start")

                    if hasattr(scaler, "mean_"):
                        print(f"   -> StandardScaler | mean={scaler.mean_[0]:.6f}, var={scaler.var_[0]:.6f}")
                    elif hasattr(scaler, "min_") and hasattr(scaler, "scale_"):
                        print(f"   -> MinMaxScaler   | min={scaler.min_[0]:.6f}, scale={scaler.scale_[0]:.6f}")
                    elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
                        print(f"   -> RobustScaler   | center={scaler.center_[0]:.6f}, scale={scaler.scale_[0]:.6f}")
                    else:
                        print(f"   -> {type(scaler).__name__} (no standard stats exposed)")

                    if dict_name in self.feature_pipeline.norm_methods:
                        method = self.feature_pipeline.norm_methods[dict_name].get(col_name, None)
                        if method:
                            print(f"   -> Normalization method in pipeline: {method}")
                    print("window end")

                seq_dict[dict_name] = df

        if printed:
            self._first_prepare_seq = False
        return seq_dict
    
def import_function(module_name, func_name):
    """
    Dynamically import a function from its module.
    """
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

def build_pipeline_from_config(cfg):
    steps = []
    for step_cfg in cfg["steps"]:
        func = import_function(step_cfg["module"], step_cfg["func"])
        steps.append(make_step(func, **step_cfg.get("kwargs", {})))

    return FeaturePipeline(
        steps=steps,
        norm_methods=cfg["norm_methods"],
        per_window_flags=cfg["per_window_flags"]
    )

def import_class(module_name, class_name):
    """Dynamically import a class by module + name"""
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
