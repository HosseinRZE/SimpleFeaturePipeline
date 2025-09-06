import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeaturePipeline:
    """
    Pipeline for preprocessing features before LSTM with multi-dict support.
    Supports per-window transformations via `per_window_flags`.
    """

    def __init__(self, steps=None, norm_methods=None, per_window_flags=None):
        self.steps = steps if steps else []
        self.norm_methods = norm_methods if norm_methods else {}
        self.scalers = {}
        # List of bools, same length as steps, indicates if a step is per-window
        self.per_window_flags = per_window_flags if per_window_flags else [False]*len(self.steps)

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
                    # clearer error: show available scalers
                    available = list(self.scalers.keys())
                    raise ValueError(
                        f"No fitted scaler found for {dict_name}.{col}. "
                        f"Available scalers: {available}"
                    )
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
            # <-- FIX: use the "main" config (not the whole dict)
            norm_cfg = self.norm_methods.get("main", {}) if isinstance(self.norm_methods, dict) else {}
            return self._normalize_single(data, norm_cfg, fit, "main")

    # --- Global transform (unchanged) ---
    def fit_transform(self, df, save_path=None):
        df_out = df.copy()
        for step in self.steps:
            df_out = step(df_out)
        df_out = self._normalize(df_out, fit=True)
        if save_path:
            joblib.dump(self.scalers, save_path)
        return df_out

    def fit(self, df):
        """
        Applies global steps and fits the scalers on the entire dataset.
        """
        df_out = df.copy()
        # Apply only global (not per-window) steps before fitting
        for step, per_win in zip(self.steps, self.per_window_flags):
            if not per_win:
                df_out = step(df_out)
    

        # Fit the normalizers on the processed data
        self._normalize(df_out, fit=True)
        return self

    def transform(self, df):
        df_out = df.copy()
        for step in self.steps:
            df_out = step(df_out)
        return self._normalize(df_out, fit=False)

    def load(self, path):
        self.scalers = joblib.load(path)

    # --- Per-window support ---
    def apply_window(self, df_window):
        """
        Apply only steps flagged as per-window transformations on a single window.
        """
        df_out = df_window.copy()
        for step, per_win in zip(self.steps, self.per_window_flags):
            if per_win:
                out = step(df_out)
                if isinstance(out, tuple):
                    df_out, _ = out
                else:
                    df_out = out
        return df_out

    def __iter__(self):
        return iter(self.steps)

    def export_config(self):
        """
        Serialize pipeline steps + normalization + flags.
        """
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