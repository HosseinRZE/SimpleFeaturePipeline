from add_ons.base_addon import BaseAddOn 
from typing import List, Dict, Any, Tuple, Literal
import pandas as pd
import numpy as np

# Define the supported processing stages
ProcessingStage = Literal["before_sequence", "apply_window"]

class CandleShapeFeaturesAddOn(BaseAddOn):
    """
    An Add-on that computes candlestick shape features.
    
    The stage can be configured to run either on the full DataFrame ('df_data')
    in before_sequence, or on individual sequences in apply_window (default).
    """
    def __init__(
        self, 
        ohlc_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
        processing_stage: ProcessingStage = "apply_window"
    ):
        """
        Initializes the add-on.

        Args:
            ohlc_cols (Tuple[str, str, str, str]): The names of the (open, high, low, close) columns.
            processing_stage (ProcessingStage): Where to run the calculation. 
                                                Default is 'apply_window'.
        """
        if processing_stage not in ["before_sequence", "apply_window"]:
            raise ValueError("processing_stage must be 'before_sequence' or 'apply_window'.")

        self.ohlc_cols = ohlc_cols
        self.processing_stage = processing_stage
        self.open_col, self.high_col, self.low_col, self.close_col = ohlc_cols

    def _calculate_shape_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core logic to calculate shape features for any given DataFrame (full or window)."""
        df_out = df.copy()

        if not all(col in df_out.columns for col in self.ohlc_cols):
            print(f"Warning: OHLC columns missing. Skipping CandleShapeFeatures calculation.")
            return df_out

        open_col, high_col, low_col, close_col = self.ohlc_cols
        
        max_oc = df_out[[open_col, close_col]].max(axis=1)
        min_oc = df_out[[open_col, close_col]].min(axis=1)
        
        # Safety for division
        max_oc_safe = max_oc.replace(0, np.nan)
        min_oc_safe = min_oc.replace(0, np.nan)
        
        # Compute features
        upper_shadow = (df_out[high_col] - max_oc) / max_oc_safe
        lower_shadow = (min_oc - df_out[low_col]) / min_oc_safe
        body = (max_oc - min_oc) / max_oc_safe
        color = np.where(
            df_out[close_col] > df_out[open_col], 0.7,
            np.where(df_out[close_col] < df_out[open_col], 0.3, 0.0)
        )
        
        # Prepare feature DataFrame
        features_df = pd.DataFrame({
            "upper_shadow": upper_shadow.fillna(0.0).astype(np.float32),
            "lower_shadow": lower_shadow.fillna(0.0).astype(np.float32),
            "body": body.fillna(0.0).astype(np.float32),
            "color": color.astype(np.float32)
        }, index=df_out.index)
        
        return pd.concat([df_out, features_df], axis=1)

    # ------------------- Option 1: Run on Full Data ------------------- #

    def before_sequence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the calculation on the full 'df_data' if processing_stage is 'before_sequence'.
        """
        if self.processing_stage == "before_sequence":
            if "df_data" in state:
                print("CandleShapeFeaturesAddOn: Running in before_sequence (Full DataFrame).")
                state["df_data"] = self._calculate_shape_features_df(state["df_data"])
            else:
                print("Warning: df_data not found. Skipping before_sequence feature generation.")
        return state

    # ------------------- Option 2: Run on Windows (Default) ------------------- #

    def apply_window(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the calculation on individual windows in 'X_list' if processing_stage is 'apply_window'.
        """
        if self.processing_stage == "apply_window":
            if "X_list" not in state:
                return state

            print("CandleShapeFeaturesAddOn: Running in apply_window (Per-Window).")
            transformed_X_list = []
            for df_window in state["X_list"]:
                if isinstance(df_window, pd.DataFrame):
                    new_df = self._calculate_shape_features_df(df_window)
                else:
                    new_df = df_window
                transformed_X_list.append(new_df)
            
            state["X_list"] = transformed_X_list

        return state