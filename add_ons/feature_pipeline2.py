import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ----------------- Updated FeaturePipeline -----------------
class FeaturePipeline:
    """
    Pipeline for preprocessing features before LSTM with multi-dict support.
    
    Usage:
        fp = FeaturePipeline(
            steps=[lambda df: add_pct_changes(df, separatable="complete")],
            norm_methods={
                "main": {"upper_shadow": "standard"},
                "pct_changes": {"open_pct": "standard", "high_pct": "standard"}
            }
        )
        df_processed = fp.fit_transform(df)
        df_new = fp.transform(df_new)

        example 2:
        pipeline = FeaturePipeline(
        steps=[
            lambda df: add_pct_changes(df, relative_to="close", separatable="no"),  # no extra dict
            lambda df: drop_columns(df, ["volume"])  # remove volume column
        ],
        norm_methods={
        "main": {"upper_shadow": "standard", "high": "minmax"}  # normalize directly in df
            }
        )
    """

    def __init__(self, steps=None, norm_methods=None):
        self.steps = steps if steps else []
        self.norm_methods = norm_methods if norm_methods else {}
        self.scalers = {}

    def _normalize_single(self, df, norm_cfg, fit, dict_name):
        """Normalize a single DataFrame according to its config."""
        df = df.copy()
        for col, method in norm_cfg.items():
            if col not in df.columns:
                continue

            col_data = df[col].values.reshape(-1, 1)

            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            elif method in (None, "none"):
                continue
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            if fit:
                col_data = scaler.fit_transform(col_data)
                self.scalers[f"{dict_name}.{col}"] = scaler
            else:
                scaler = self.scalers.get(f"{dict_name}.{col}")
                if scaler is None:
                    raise ValueError(f"No fitted scaler found for {dict_name}.{col}")
                col_data = scaler.transform(col_data)

            df[col] = col_data.flatten()
        return df

    def _normalize(self, data, fit=True):
        """Normalize either a single DataFrame or a dict of DataFrames."""
        if isinstance(data, dict):
            out = {}
            for dict_name, df in data.items():
                norm_cfg = self.norm_methods.get(dict_name, {})
                out[dict_name] = self._normalize_single(df, norm_cfg, fit, dict_name)
            return out
        else:
            return self._normalize_single(data, self.norm_methods, fit, "main")

    def fit_transform(self, df, save_path=None):
        """Apply feature steps, fit scalers, and transform."""
        df_out = df.copy()
        for step in self.steps:
            df_out = step(df_out)
        df_out = self._normalize(df_out, fit=True)
        if save_path:
            joblib.dump(self.scalers, save_path)
        return df_out

    def transform(self, df):
        """Apply feature steps and transform using fitted scalers."""
        df_out = df.copy()
        for step in self.steps:
            df_out = step(df_out)
        return self._normalize(df_out, fit=False)

    def load(self, path):
        """Load previously saved scalers."""
        self.scalers = joblib.load(path)

    def __iter__(self):
        """Make pipeline iterable over steps (for preprocess_csv)."""
        return iter(self.steps)
