import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib

class FeaturePipeline:
    def __init__(self, steps=None, per_window_flags=None, norm_methods=None):
        self.steps = steps or []
        self.per_window_flags = per_window_flags or [False] * len(self.steps)
        self.norm_methods = norm_methods or {}
        self.scalers = {}            # Store sklearn scalers for each column
        self.global_dicts = {}       # Store dicts created by global steps
        self.global_invalid = set()  # Store invalid indices from global steps
        self.main_data = None        # Store main DataFrame after global processing

    # -------------------
    # Step application
    # -------------------
    def _apply_step(self, df, dicts, step):
        """Apply a single step and handle optional output formats."""
        result = step(df)
        if isinstance(result, tuple):
            if len(result) == 2:  # (df, bad_indices)
                return result[0], result[1], dicts
            elif len(result) == 3:  # (df, bad_indices, dicts_new)
                dicts.update(result[2])
                return result[0], result[1], dicts
        return result, set(), dicts

    # -------------------
    # Global fit
    # -------------------
    def fit(self, df):
        """Fit global steps and normalization."""
        df_out = df.copy()
        dicts = {}
        global_bad = set()  # collect all global bad indices

        # --- Apply global steps ---
        for step, per_win in zip(self.steps, self.per_window_flags):
            if not per_win:
                df_out, bad, dicts_new = self._apply_step(df_out, dicts, step)
                dicts.update(dicts_new)
                global_bad.update(bad)

        # Save results
        self.global_dicts = dicts
        self.global_invalid = global_bad
        self.global_bad_indices = global_bad  # <-- new attribute
        self.main_data = df_out.copy()

        # --- Fit normalization ---
        data_all = {"main": df_out, **dicts}
        self._normalize(data_all, fit=True)

        return self

    # -------------------
    # Apply per-window
    # -------------------
    def apply_window(self, dicts):
        """
        Apply per-window steps and normalization.
        Sequences containing any global_bad_indices are skipped (return None).
        """
        # If sequence contains any global bad indices, skip
        if any(idx in self.global_bad_indices for idx in dicts["main"].index):
            return None

        dicts_out = {k: v.copy() for k, v in dicts.items()}

        for step, per_win in zip(self.steps, self.per_window_flags):
            if not per_win:
                continue
            result = step(dicts_out["main"])
            if len(result) == 2:
                df_sub, bad_idx = result
                if len(bad_idx) > 0:
                    return None  # skip sequence if per-window step fails
                dicts_out["main"] = df_sub
            elif len(result) == 3:
                df_sub, bad_idx, dicts_new = result
                if len(bad_idx) > 0:
                    return None
                dicts_out["main"] = df_sub
                for k, v in dicts_new.items():
                    dicts_out[k] = v
            else:
                raise ValueError("Step must return 2- or 3-tuple")

        return dicts_out

    # -------------------
    # Normalization
    # -------------------
    def _normalize_single(self, df, norm_cfg, fit, dict_name):
        df_out = df.copy()
        for col, method in norm_cfg.items():
            if col not in df_out.columns:
                continue

            col_data = df_out[col].values.reshape(-1, 1)
            scaler_key = f"{dict_name}.{col}"

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
                self.scalers[scaler_key] = scaler
            else:
                scaler = self.scalers.get(scaler_key)
                if scaler is None:
                    raise ValueError(f"No fitted scaler for {scaler_key}")
                col_data = scaler.transform(col_data)

            df_out[col] = col_data.flatten()
        return df_out

    def _normalize(self, data, fit=True):
        """Normalize either a dict of DataFrames or a single DataFrame."""
        if isinstance(data, dict):
            for dict_name, df in data.items():
                norm_cfg = self.norm_methods.get(dict_name, {})
                data[dict_name] = self._normalize_single(df, norm_cfg, fit, dict_name)
        else:
            data = self._normalize_single(data, self.norm_methods.get("main", {}), fit, "main")
        return data

    # -------------------
    # Extra utilities
    # -------------------
    def __iter__(self):
        return iter(self.steps)

    def export_config(self):
        """Serialize pipeline steps + normalization + flags."""
        steps_cfg = []
        for step in self.steps:
            steps_cfg.append({
                "module": getattr(step, "_step_module", None),
                "func": getattr(step, "_step_name", step.__name__),
                "kwargs": getattr(step, "_step_kwargs", {})
            })
        return {
            "steps": steps_cfg,
            "norm_methods": self.norm_methods,
            "per_window_flags": self.per_window_flags
        }

    def fit_y(self, df_labels, lineprice_cols):
        """
        Fit target scalers on label columns, ignoring NaN and zero placeholders.
        """
        for col in lineprice_cols:
            all_vals = df_labels[col].values.astype(np.float32).flatten()
            mask = ~np.isnan(all_vals) & (all_vals != 0)
            if mask.sum() > 0:
                scaler = StandardScaler().fit(all_vals[mask].reshape(-1, 1))
                self.target_scalers[col] = scaler
        return self

    def transform_y(self, df_labels, lineprice_cols):
        """
        Apply fitted target scalers to label columns.
        Only masked values (non-NaN, non-zero) are transformed.
        """
        df_out = df_labels.copy()
        for col in lineprice_cols:
            if col not in self.target_scalers:
                continue
            scaler = self.target_scalers[col]
            vals = df_out[col].values.astype(np.float32)
            mask = ~np.isnan(vals) & (vals != 0)
            if mask.sum() > 0:
                vals[mask] = scaler.transform(vals[mask].reshape(-1, 1)).flatten()
            df_out[col] = vals
        return df_out

    def export_target_config(self):
        """
        Export target scaler configuration as a dict
        (to be embedded in metadata).
        """
        return {
            name: {
                "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
                "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
                "class": scaler.__class__.__name__
            }
            for name, scaler in self.target_scalers.items()
    }

    def load(self, path):
        """Load fitted scalers from disk."""
        self.scalers = joblib.load(path)
