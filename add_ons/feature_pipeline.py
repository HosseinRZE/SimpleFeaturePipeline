from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import numpy as np
import joblib

# ----------------- New FeaturePipeline -----------------
class FeaturePipeline:
    """
    Pipeline for preprocessing features before LSTM.
    
    This class helps structure feature engineering and normalization steps 
    applied to a DataFrame before converting it into sequences for the LSTM.

    Usage:
        1. Create a pipeline:
            fp = FeaturePipeline()

        2. Add feature engineering steps (run once on the raw DataFrame):
            fp.add_step(lambda df: add_zigzag(df, window_size=3, dev_threshold=1))
            fp.add_step(lambda df: drop_columns(df, ["open", "high"]))

        3. (Optional) Specify normalization methods for columns:
            fp.set_normalization({
                "zigzag": "standard",   # standardize zigzag values
                "rsi": "minmax",        # min-max normalize RSI
                "volume": None          # leave volume unchanged
            })

        4. Apply pipeline:
            df_processed = fp.fit_transform(df)  # fits scalers + applies steps
            df_new       = fp.transform(df_new) # only transforms (uses fitted scalers)

    Methods:
        - add_step(func):
            Add a preprocessing function (df -> df).
            Examples: `add_zigzag`, `drop_columns`, technical indicators, etc.
            Use this to shape your features BEFORE normalization.

        - set_normalization(norm_dict):
            Define per-column normalization strategies.
            Allowed values: "standard", "minmax", None
            Example:
                {"zigzag": "standard", "rsi": "minmax", "volume": None}

        - fit_transform(df):
            Apply feature steps, then fit scalers on columns, then transform.
            Use this during training data preprocessing.

        - transform(df):
            Apply feature steps, then use previously fitted scalers to transform.
            Use this during validation/test preprocessing to avoid data leakage.

        - __iter__():
            Makes the pipeline iterable over steps, so it can be used in 
            existing code expecting a list of functions (e.g. preprocess_csv).

    Notes:
        - Always call `fit_transform` on your training data.
        - Call `transform` on validation/test sets.
        - Normalization is optional; if not set, no normalization is applied.
    """

    def __init__(self, steps=None, norm_methods=None):
        self.steps = steps if steps else []
        self.norm_methods = norm_methods if norm_methods else {}
        self.scalers = {}

    def normalize_columns(df, col_methods):
        """
        Normalize or transform columns in a DataFrame with specified methods.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        col_methods : dict
            Dictionary mapping column names to a list of transformations.
            Example:
                {
                    "volume": ["log", "standard"],
                    "price_change": ["robust"],
                    "feature_x": ["minmax"]
                }

        Supported methods
        -----------------
        - "log" : log(x + 1)
        - "standard" : zero mean, unit variance
        - "minmax" : scale to [0,1]
        - "robust" : scale using median/IQR
        - "none" : leave unchanged
        """
        df = df.copy()

        for col, methods in col_methods.items():
            if col not in df.columns:
                continue

            col_data = df[col].values.reshape(-1, 1)
            for method in methods:
                if method == "log":
                    col_data = np.log1p(col_data)
                elif method == "standard":
                    scaler = StandardScaler()
                    col_data = scaler.fit_transform(col_data)
                elif method == "minmax":
                    scaler = MinMaxScaler()
                    col_data = scaler.fit_transform(col_data)
                elif method == "robust":
                    scaler = RobustScaler()
                    col_data = scaler.fit_transform(col_data)
                elif method == "none":
                    pass
                else:
                    raise ValueError(f"Unknown normalization method: {method}")

            df[col] = col_data.flatten()

        return df

    def fit_transform(self, df, save_path=None):
        df_out = df.copy()
        for step in self.steps:
            df_out = step(df_out)
        df_out = self._normalize(df_out, fit=True)
        if save_path:
            joblib.dump(self.scalers, save_path)
        return df_out

    def transform(self, df):
        df_out = df.copy()
        for step in self.steps:
            df_out = step(df_out)
        return self._normalize(df_out, fit=False)

    def load(self, path):
        self.scalers = joblib.load(path)

    def __iter__(self):
        return iter(self.steps) 
