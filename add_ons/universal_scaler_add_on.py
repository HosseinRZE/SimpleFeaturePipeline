import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from add_ons.base_addon import BaseAddOn 


class UniversalScalerAddOn(BaseAddOn):
    """
    An Add-on that performs ALL operations (fitting and transforming) 
    within the apply_window stage. It fits on the original df_data/df_labels 
    and then transforms X_list/y_list.
    """
    def __init__(self, feature_columns: List[str], label_columns: List[str] = None):
        """
        Args:
            feature_columns (list): List of feature column names to scale.
            label_columns (list): List of label column names to scale.
        """
        if not feature_columns:
            raise ValueError("Feature columns must be specified for UniversalScalerAddOn.")
            
        self.feature_columns = feature_columns
        self.label_columns = label_columns if label_columns is not None else []
        self.universal_scaler: StandardScaler = None

    # ------------------- Data Processing Stages ------------------- #

    def before_sequence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pass-through, as all logic is now handled in apply_window as requested.
        """
        return state

    def _fit_scaler(self, df_data: pd.DataFrame, df_labels: pd.DataFrame) -> None:
        """Helper to encapsulate the fitting logic."""
        
        # Default label columns if not explicitly set
        if not self.label_columns:
            self.label_columns = [col for col in df_labels.columns if col.startswith("linePrice")]

        # 1. Extract and Flatten Values
        try:
            all_feature_values = df_data[self.feature_columns].values.flatten()
            all_label_values = df_labels[self.label_columns].values.flatten()
        except KeyError as e:
            raise ValueError(f"Column not found in DataFrame during fitting: {e}")

        # 2. Combine and Fit the Scaler
        combined_values = np.concatenate([all_feature_values, all_label_values])
        
        if combined_values.size == 0:
            print("Warning: Combined data is empty. Skipping scaler fitting.")
            return

        self.universal_scaler = StandardScaler()
        # Reshape to (N_samples, 1) as required by scikit-learn
        self.universal_scaler.fit(combined_values.reshape(-1, 1))

        print("UniversalScalerAddOn: Scaler fitted on combined data and labels (within apply_window).")


    def apply_window(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. FITS the scaler if it hasn't been fitted yet (using df_data/df_labels).
        2. TRANSFORMS individual windows in X_list and y_list.
        """
        
        # --- Step 1: Fitting (Only happens once) ---
        if self.universal_scaler is None:
            if "df_data" in state and "df_labels" in state:
                self._fit_scaler(state["df_data"], state["df_labels"])
                # Store the scaler in state for reference if needed
                state["universal_scaler"] = self.universal_scaler
                
                # Optional: Remove large DataFrames from state after use to free memory
                # del state["df_data"]
                # del state["df_labels"]
            else:
                # Cannot fit if full data is missing
                return state 

        if "X_list" not in state or self.universal_scaler is None:
            return state
        
        scaler = self.universal_scaler
        
        # Helper function to transform a list of DataFrames
        def transform_df_list(df_list: List[pd.DataFrame], columns: List[str]) -> List[pd.DataFrame]:
            transformed_list = []
            for df_window in df_list:
                df_window_copy = df_window.copy()
                cols_to_scale = [col for col in columns if col in df_window_copy.columns]

                if cols_to_scale:
                    # Extract, flatten, transform, and reshape back
                    original_values = df_window_copy[cols_to_scale].values
                    original_shape = original_values.shape
                    
                    # Reshape to (N_timesteps * N_features, 1) and transform
                    transformed_values_flat = scaler.transform(original_values.reshape(-1, 1))
                    
                    # Reshape back to the original shape
                    transformed_values = transformed_values_flat.reshape(original_shape)
                    
                    df_window_copy[cols_to_scale] = transformed_values
                
                transformed_list.append(df_window_copy)
            return transformed_list

        # --- Step 2: Transformation ---
        
        # 1. Transform X_list
        state["X_list"] = transform_df_list(state["X_list"], self.feature_columns)

        # 2. Transform y_list 
        if "y_list" in state and self.label_columns:
             if all(isinstance(y, pd.DataFrame) for y in state["y_list"]):
                 state["y_list"] = transform_df_list(state["y_list"], self.label_columns)

        return state

    # ------------------- Evaluation Stages ------------------- #

    def on_evaluation(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inverse scales predictions and labels using the universal scaler.
        """
        if self.universal_scaler is None:
            return eval_data

        all_preds = eval_data.get("all_preds_reg")
        all_labels = eval_data.get("all_labels_reg")

        if all_preds is not None:
            original_shape = all_preds.shape
            
            # Reshape to (N_samples * Timesteps * Features, 1) for the universal scaler
            all_preds_flat = all_preds.flatten().reshape(-1, 1)
            all_labels_flat = all_labels.flatten().reshape(-1, 1)
            
            # Apply the inverse transform
            eval_data["all_preds_reg"] = self.universal_scaler.inverse_transform(all_preds_flat).reshape(original_shape)
            eval_data["all_labels_reg"] = self.universal_scaler.inverse_transform(all_labels_flat).reshape(original_shape)
        
        return eval_data