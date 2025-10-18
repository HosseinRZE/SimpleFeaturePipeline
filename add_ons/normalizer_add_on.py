from add_ons.base_addon import BaseAddOn 
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define the supported normalization methods
NormMethod = Literal["standard", "minmax", "none"]

class ColumnNormalizer:
    """
    A class to manage and apply column-wise normalization using scikit-learn scalers.
    """
    def __init__(self, norm_map: Dict[str, NormMethod]):
        self.norm_map = {k: v.lower() for k, v in norm_map.items()}
        self.scalers: Dict[str, Any] = {}
        self.supported_methods = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
        }

    def fit(self, df: pd.DataFrame):
        """Fits the necessary scalers to the provided DataFrame."""
        self.scalers = {}
        for col, method in self.norm_map.items():
            if method in self.supported_methods:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found for fitting.")
                    continue

                ScalerClass = self.supported_methods[method]
                # Scaler expects 2D array, so we pass df[[col]]
                scaler = ScalerClass().fit(df[[col]])
                self.scalers[col] = scaler
        return self

    def transform_window(self, df_window: pd.DataFrame) -> pd.DataFrame:
        """
        Applies transformation to a single DataFrame window/subsequence.
        This is called repeatedly in apply_window.
        """
        df_transformed = df_window.copy()
        
        for col, scaler in self.scalers.items():
            if col in df_window.columns:
                # Apply transformation to the column in the current window
                df_transformed[col] = scaler.transform(df_transformed[[col]]).flatten()
            
        return df_transformed
    
    # Keeping inverse_transform for completeness during evaluation
    def inverse_transform(self, arr: np.ndarray, col_name: str) -> np.ndarray:
        # ... (Inverse transform logic from previous answer) ...
        if col_name in self.scalers:
            scaler = self.scalers[col_name]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return scaler.inverse_transform(arr).flatten()
        return arr.flatten()
    
class ColumnNormalizationAddOn(BaseAddOn):
    """
    An Add-on that fits scalers on the full dataset (before_sequence) and
    applies the transformation to individual windows (apply_window).
    """
    def __init__(self, data_norm_map: Dict[str, NormMethod], label_norm_map: Dict[str, NormMethod] = None):
        """
        Args:
            data_norm_map: Dict mapping data column names to normalization methods.
            label_norm_map: Optional dict mapping label column names to normalization methods.
        """
        self.data_norm_map = data_norm_map
        self.label_norm_map = label_norm_map if label_norm_map is not None else {}
        self.data_normalizer: ColumnNormalizer = None
        self.label_normalizer: ColumnNormalizer = None

    def before_sequence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fits scalers to df_data and df_labels (if available and requested).
        The dataframes themselves are NOT transformed here.
        """
        if "df_data" not in state:
            return state

        df_data = state["df_data"]
        df_labels = state.get("df_labels", pd.DataFrame())

        # 1. Fit Data Columns
        if self.data_norm_map:
            # Fit the normalizer on the full training data
            self.data_normalizer = ColumnNormalizer(self.data_norm_map).fit(df_data)
            state["data_normalizer"] = self.data_normalizer
            
        # 2. Fit Label Columns (If normalization is requested)
        if self.label_norm_map and not df_labels.empty:
            # Fit the normalizer on the full training labels
            self.label_normalizer = ColumnNormalizer(self.label_norm_map).fit(df_labels)
            state["label_normalizer"] = self.label_normalizer

        print("ColumnNormalizationAddOn: Scalers fitted successfully.")
        return state

    def apply_window(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the column normalization transformation to each window in X_list and y_list.
        """
        if "X_list" not in state or self.data_normalizer is None:
            return state
        
        # --- Transform X_list (Data) ---
        transformed_X_list = []
        for df_window in state["X_list"]:
            # Apply the fitted scaler to the current window
            new_df = self.data_normalizer.transform_window(df_window)
            transformed_X_list.append(new_df)
        state["X_list"] = transformed_X_list

        # --- Transform y_list (Labels) ---
        if "y_list" in state and self.label_normalizer is not None:
            transformed_y_list = []
            for df_label_window in state["y_list"]:
                # Apply the fitted scaler to the current label window
                # Note: y_list might contain simple arrays/series, but we assume it matches
                # the structure of df_labels for column-wise scaling.
                if isinstance(df_label_window, pd.DataFrame):
                     new_df = self.label_normalizer.transform_window(df_label_window)
                elif isinstance(df_label_window, (pd.Series, np.ndarray)) and len(self.label_norm_map) == 1:
                    # Special case for single-column labels/series
                    col_name = list(self.label_norm_map.keys())[0]
                    df_temp = pd.DataFrame({col_name: df_label_window})
                    new_df = self.label_normalizer.transform_window(df_temp)
                    new_df = new_df[col_name] # Return as series/array-like
                else:
                    new_df = df_label_window # Skip transformation

                transformed_y_list.append(new_df)

            state["y_list"] = transformed_y_list

        return state

    def on_evaluation(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inverse scales predictions and labels if they were normalized. (Same as before)
        """
        # ... (Inverse transform logic from previous answer, using self.label_normalizer) ...
        if self.label_normalizer is None:
            return eval_data

        all_preds = eval_data.get("all_preds_reg")
        all_labels = eval_data.get("all_labels_reg")
        label_cols = list(self.label_norm_map.keys())

        if all_preds is not None and len(label_cols) > 0:
            original_shape = all_preds.shape
            N_features = all_preds.shape[-1]
            
            all_preds_2d = all_preds.reshape(-1, N_features)
            all_labels_2d = all_labels.reshape(-1, N_features)
            
            for i, col_name in enumerate(label_cols):
                if i < N_features:
                    all_preds_2d[:, i] = self.label_normalizer.inverse_transform(
                        all_preds_2d[:, i], col_name=col_name
                    )
                    all_labels_2d[:, i] = self.label_normalizer.inverse_transform(
                        all_labels_2d[:, i], col_name=col_name
                    )
            
            eval_data["all_preds_reg"] = all_preds_2d.reshape(original_shape)
            eval_data["all_labels_reg"] = all_labels_2d.reshape(original_shape)
        
        return eval_data