import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib

class FeaturePipeline:
    def __init__(self, steps=None, per_window_flags=None, norm_methods=None, window_norms=None):
        self.steps = steps or []
        self.per_window_flags = per_window_flags or [False] * len(self.steps)
        self.norm_methods = norm_methods or {}
        self.scalers = {}            # Store sklearn scalers for each column
        self.global_dicts = {}       # Store dicts created by global steps
        self.global_invalid = set()  # Store invalid indices from global steps
        self.main_data = None        # Store main DataFrame after global processing
        self.target_scaler = {} 
        self.window_scalers = {} 
        self.window_norms = window_norms or {}

    def _make_scaler(self, method):
        if method == "standard":
            return StandardScaler()
        if method == "robust":
            return RobustScaler()
        if method == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unknown normalization method: {method}")
    
    def export_target_scalers(self):
        return self.target_scaler

    def load_target_scalers(self, scalers_dict):
        self.target_scaler = scalers_dict
        return self.target_scaler
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
    def fit(self, df, normalize = True):
        df_out = df.copy()
        dicts = {}
        global_bad = set()

        # --- Apply global steps ---
        for step, per_win in zip(self.steps, self.per_window_flags):
            if not per_win:
                df_out, bad, dicts_new = self._apply_step(df_out, dicts, step)
                dicts.update(dicts_new)
                global_bad.update(bad)

        # Save results
        self.global_dicts = dicts
        self.global_invalid = global_bad
        self.global_bad_indices = global_bad
        self.main_data = df_out.copy()

        # --- Fit normalization ---
        data_all = {"main": df_out, **dicts}
        # we do not assign the output so tha only save scalars for later use
        if normalize ==True:
            self._normalize(data_all, fit=True)

        # --- Return a single dict for downstream slicing ---
        self.global_data = data_all  # <-- new attribute
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


    def apply_window_normalization(self, X_dicts_list, feature_columns, fit=True):
        """
        Apply per-column window normalization across all collected windows.

        - X_dicts_list: list of dicts, each dict_name -> numpy array (T_i, n_features) for that sample
        - feature_columns: dict mapping dict_name -> list of column names (order must match arrays)
        - fit: if True, fit scalers from the concatenated column values; if False, just transform using already-fitted scalers.
        
        Stores fitted scalers in self.window_scalers with keys (dict_name, col_name).
        Returns the modified X_dicts_list (modifies in place).
        """
        if not self.window_norms:
            return X_dicts_list

        for dict_name, col_method_map in self.window_norms.items():
            cols = feature_columns.get(dict_name)
            if cols is None:
                # no feature names captured for this dict -> skip
                continue

            # For each column configured to be window-normalized
            for col_name, method in col_method_map.items():
                # determine column index
                try:
                    # prefer resolving by name
                    col_idx = cols.index(col_name)
                except ValueError:
                    # if user supplied an integer-like key, try to interpret it
                    try:
                        col_idx = int(col_name)
                    except Exception:
                        # skip if cannot resolve
                        continue

                # collect values across all samples for this column
                pieces = []
                for sample in X_dicts_list:
                    arr = sample.get(dict_name)
                    if arr is None:
                        continue
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.size == 0:
                        continue
                    # handle 1D arrays (single feature) vs 2D
                    if arr.ndim == 1:
                        if col_idx != 0:
                            continue
                        pieces.append(arr.reshape(-1, 1))
                    else:
                        if col_idx >= arr.shape[1]:
                            continue
                        pieces.append(arr[:, col_idx].reshape(-1, 1))

                if not pieces:
                    # nothing to fit/transform for this column
                    continue

                all_vals = np.vstack(pieces)  # shape (total_timesteps_across_all_windows, 1)
                key = (dict_name, col_name)

                if fit:
                    scaler = self._make_scaler(method)
                    scaler.fit(all_vals)
                    self.window_scalers[key] = scaler
                else:
                    scaler = self.window_scalers.get(key)
                    if scaler is None:
                        raise RuntimeError(
                            f"No fitted window scaler found for {key}. Call with fit=True first (on training data)."
                        )

                # transform each sample in-place
                for sample in X_dicts_list:
                    arr = sample.get(dict_name)
                    if arr is None:
                        continue
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.size == 0:
                        continue

                    if arr.ndim == 1:
                        if col_idx != 0:
                            continue
                        transformed = scaler.transform(arr.reshape(-1, 1)).reshape(-1)
                        sample[dict_name] = transformed.astype(np.float32)
                    else:
                        if col_idx >= arr.shape[1]:
                            continue
                        col_vals = arr[:, col_idx].reshape(-1, 1)
                        arr[:, col_idx] = scaler.transform(col_vals).reshape(-1)
                        sample[dict_name] = arr.astype(np.float32)

        return X_dicts_list

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
        Fit ONE target scaler across all linePrice columns, ignoring NaN and zeros.
        """
        all_vals = df_labels[lineprice_cols].values.astype(np.float32).flatten()
        mask = ~np.isnan(all_vals) & (all_vals != 0)
        if mask.sum() > 0:
            scaler = StandardScaler().fit(all_vals[mask].reshape(-1, 1))
            self.target_scaler = scaler

        return self

    def transform_y(self, df_labels, lineprice_cols):
        df_out = df_labels.copy()
        if hasattr(self, "target_scaler"):
            for col in lineprice_cols:
                vals = df_out[col].values.astype(np.float32)
                mask = ~np.isnan(vals) & (vals != 0)
                if mask.sum() > 0:
                    vals[mask] = self.target_scaler.transform(vals[mask].reshape(-1, 1)).flatten()
                vals = np.nan_to_num(vals, nan=0.0)
                df_out[col] = vals
        return df_out
    
    def load(self, path):
        """Load fitted scalers from disk."""
        self.scalers = joblib.load(path)

