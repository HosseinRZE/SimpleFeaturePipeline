import pandas as pd 
import importlib
from utils.make_step import make_step
from add_ons.feature_pipeline3 import FeaturePipeline
class ServerPreprocess:
    def __init__(self, feature_pipeline, scalers_path=None):
        """
        Stateful preprocessing for serving.
        - feature_pipeline: same FeaturePipeline used in training
        - scalers_path: path to saved scalers (from training)
        """
        self.feature_pipeline = feature_pipeline
        self.dataset = pd.DataFrame()

        # Load pre-fitted scalers from training
        if scalers_path is not None:
            self.feature_pipeline.load(scalers_path)

    def add_candle(self, new_candle: pd.Series | dict):
        """
        Add one new candle (dict or Series) to internal dataset.
        Applies only the global (non per-window) pipeline steps.
        """
        if isinstance(new_candle, dict):
            new_candle = pd.DataFrame([new_candle])
        elif isinstance(new_candle, pd.Series):
            new_candle = pd.DataFrame([new_candle.to_dict()])
        else:
            raise ValueError("new_candle must be dict or pd.Series")

        # Apply only global steps
        for step, per_window in zip(self.feature_pipeline.steps, self.feature_pipeline.per_window_flags):
            if not per_window:
                new_candle = step(new_candle)

        # Append to dataset
        self.dataset = pd.concat([self.dataset, new_candle], ignore_index=True)

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
