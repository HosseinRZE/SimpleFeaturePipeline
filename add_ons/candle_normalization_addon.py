from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from add_ons.base_addon import BaseAddOn 

# Re-defining the utility function (can be kept outside the class)
def add_label_normalized_candles(
    df: pd.DataFrame,
    ohlc_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
    suffix: str = "_prop",
) -> pd.DataFrame:
    """
    For a single subsequence (window), compute per-candle proportions:
    each candle's OHLC divided by the window's *last* close.

    Adds 4 columns to df: open_prop, high_prop, low_prop, close_prop (float32).
    """
    df = df.copy()

    if df.empty:
        for c in ohlc_cols:
            df[f"{c}{suffix}"] = 0.0
        return df

    # Get the last 'close' value of the window
    last_close = df["close"].iloc[-1]
    
    # Handle division by zero or non-finite values (like NaN/Inf)
    if not np.isfinite(last_close) or last_close == 0.0:
        for c in ohlc_cols:
            df[f"{c}{suffix}"] = 0.0
        return df

    # Compute the proportions
    for c in ohlc_cols:
        # Note: Added .values for explicit array-based division for speed 
        # and ensuring correct dtype assignment
        df[f"{c}{suffix}"] = (df[c].values.astype(np.float32) / last_close)

    return df

# --- Concrete Add-On Class ---
class CandleNormalizationAddOn(BaseAddOn):    
    """
    An Add-on that computes per-candle OHLC proportions normalized
    by the last 'close' price of the sequence (window).
    """
    def __init__(
        self, 
        ohlc_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"), 
        suffix: str = "_prop",
        feature_group_key: str = "main"  # <-- ADD THIS: Key for the target DataFrame
    ):
        """
        Initializes the add-on with column names, suffix, and the dictionary key.
        """
        self.ohlc_cols = ohlc_cols
        self.suffix = suffix
        self.feature_group_key = feature_group_key

    def apply_window(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the label-normalized candle transformation to each sequence (window).
        """
        if "X_list" not in state:
            return state

        transformed_X_list = []
        # --- MODIFIED LOGIC STARTS HERE ---
        for x_sample_dict in state["X_list"]:
            # x_sample_dict is a dictionary like {'main': df}
            
            # 1. Get the actual DataFrame from the dictionary
            main_df = x_sample_dict[self.feature_group_key]
            
            # 2. Apply the transformation to that DataFrame
            transformed_df = add_label_normalized_candles(
                df=main_df,
                ohlc_cols=self.ohlc_cols,
                suffix=self.suffix
            )
            
            # 3. Create a copy of the sample dict and update it with the transformed DataFrame
            new_sample_dict = x_sample_dict.copy()
            new_sample_dict[self.feature_group_key] = transformed_df
            
            # 4. Append the updated dictionary to our results
            transformed_X_list.append(new_sample_dict)
        # --- MODIFIED LOGIC ENDS HERE ---
        
        state["X_list"] = transformed_X_list
        return state