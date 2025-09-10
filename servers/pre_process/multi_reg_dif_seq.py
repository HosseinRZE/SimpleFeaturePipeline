import pandas as pd 
import importlib
from utils.make_step import make_step
from add_ons.feature_pipeline3 import FeaturePipeline
import numpy as np
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

        # Load pre-fitted scalers from training
        if scalers_path is not None:
            self.feature_pipeline.load(scalers_path)

    def add_candle(self, new_candle: pd.Series | dict):
        """
        Add one new candle (dict or Series) to internal dataset.
        Applies only the global (non per-window) pipeline steps.
        Also stores raw candle in reference_dataset.
        """
        if isinstance(new_candle, dict):
            new_candle_df = pd.DataFrame([new_candle])
        elif isinstance(new_candle, pd.Series):
            new_candle_df = pd.DataFrame([new_candle.to_dict()])
        else:
            raise ValueError("new_candle must be dict or pd.Series")

        # 👉 keep a copy of the raw OHLC before transforming
        self.add_reference_candle(new_candle_df)

        # Apply only global steps
        proc_candle = new_candle_df.copy()
        for step, per_window in zip(self.feature_pipeline.steps, self.feature_pipeline.per_window_flags):
            if not per_window:
                proc_candle = step(proc_candle)
                # 🔑 Handle case when step returns a tuple
                if isinstance(proc_candle, tuple):
                    proc_candle = proc_candle[0]

        # Append to processed dataset
        self.dataset = pd.concat([self.dataset, proc_candle], ignore_index=True)


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
        """
        Get the last `seq_len` rows from dataset,
        apply per-window steps and normalization (using training scalers).
        """
        if len(self.dataset) < seq_len:
            raise ValueError(f"Not enough data: have {len(self.dataset)}, need {seq_len}")

        seq_df = self.dataset.iloc[-seq_len:].copy()

        # Apply per-window steps
        seq_df = self.feature_pipeline.apply_window(seq_df)

        # Apply normalization using pre-fitted scalers
        seq_df = self.feature_pipeline._normalize(seq_df, fit=False)

        return seq_df

    def prepare_xgboost_seq(self, seq_len: int, model=None) -> np.ndarray:
        """
        Prepare a sequence for XGBoost prediction.
        - Takes last `seq_len` rows from dataset
        - Applies per-window steps and normalization
        - Average-pools across time
        - Optionally filters to model's expected features
        - Returns flat NumPy vector (shape: [1, n_features])
        """
        if len(self.dataset) < seq_len:
            raise ValueError(f"Not enough data: have {len(self.dataset)}, need {seq_len}")

        seq_df = self.dataset.iloc[-seq_len:].copy()

        # Apply per-window steps
        seq_df = self.feature_pipeline.apply_window(seq_df)

        # Apply normalization using pre-fitted scalers
        seq_df = self.feature_pipeline._normalize(seq_df, fit=False)

        # --- If model is provided, align features ---
        if model is not None:
            if hasattr(model, "feature_names_in_"):  # sklearn API
                expected = list(model.feature_names_in_)
            else:  # fallback: just number of features
                expected = seq_df.columns[: model.get_booster().num_features()]
            seq_df = seq_df[expected]

        # Average pool across time (axis=0)
        feat_vec = seq_df.mean(axis=0).to_numpy(dtype=np.float32)

        return feat_vec.reshape(1, -1)

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
