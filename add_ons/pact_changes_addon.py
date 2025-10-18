from add_ons.base_addon import BaseAddOn 
from typing import List, Dict, Any, Literal
import pandas as pd
import numpy as np

# Define the supported calculation methods
RelativeTo = Literal["same", "close"]

class PctChangesAddOn(BaseAddOn):
    """
    An Add-on that calculates OHLC percentage change features within each sequence (window).
    The new columns are merged directly into the feature window DataFrame.
    
    This operation runs after the data has been sequenced (in apply_window).

    Examples:
    >>> # 1. Calculate % change vs same field in the previous candle (standard OHLC pct_change)
    >>> same_pct_addon = PctChangesAddOn(relative_to="same")
    
    >>> # 2. Calculate all OHLC fields relative to the previous candle's close price
    >>> close_pct_addon = PctChangesAddOn(relative_to="close")
    """
    def __init__(self, relative_to: RelativeTo = "same"):
        """
        Initializes the add-on.

        Args:
            relative_to (RelativeTo): Determines the basis for the percentage change calculation.
                - "same": % change vs same field in previous candle (df.pct_change()).
                - "close": All OHLC fields relative to the previous candle's close price.
        
        Raises:
            ValueError: If relative_to is not 'same' or 'close'.
        """
        if relative_to not in ["same", "close"]:
            raise ValueError("relative_to must be 'same' or 'close'")
            
        self.relative_to = relative_to
        self.ohlc_cols = ["open", "high", "low", "close"]
        
    def _calculate_pct_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal helper to calculate percentage changes for a single window."""
        df_out = df.copy()

        # Check for essential columns
        if not all(col in df_out.columns for col in self.ohlc_cols):
            print(f"Warning: OHLC columns missing in window. Skipping PctChanges calculation.")
            return df_out

        # Handle NaNs that occur in the first row after shifting/pct_change
        fill_value = 0.0

        if self.relative_to == "same":
            for col in self.ohlc_cols:
                df_out[f"{col}_pct"] = df_out[col].pct_change().fillna(fill_value)
                
        elif self.relative_to == "close":
            # Shift the 'close' price to get the previous candle's close
            prev_close = df_out["close"].shift(1)
            
            # The first value of prev_close is NaN. For the first candle, 
            # use the candle's own close (i.e., calculate 0% change).
            prev_close = prev_close.fillna(df_out["close"].iloc[0])

            # Calculate % change for all OHLC relative to the previous close
            for col in self.ohlc_cols:
                # pct_change = (current_price - prev_close) / prev_close
                df_out[f"{col}_pct"] = ((df_out[col] - prev_close) / prev_close).fillna(fill_value)
                
        return df_out

    def apply_window(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the percentage change calculation to each DataFrame in X_list.
        
        Args:
            state (Dict): Contains 'X_list' (list of pd.DataFrame windows).
        Returns:
            Dict: Updated state with transformed X_list.

        Examples:
        >>> # Setup dummy windows
        >>> w1 = pd.DataFrame({'open': [10, 11], 'close': [12, 13]})
        >>> w2 = pd.DataFrame({'open': [5, 6], 'close': [7, 8]})
        >>> initial_state = {'X_list': [w1.copy(), w2.copy()]}
        
        >>> # Initialize addon for 'same' relative_to
        >>> addon = PctChangesAddOn(relative_to="same")
        
        >>> # Run the stage
        >>> updated_state = addon.apply_window(initial_state)
        
        >>> # Verify the first window's new column: [0.0 (nan filled), (13-12)/12]
        >>> # For w1.close_pct: [0.0, (13-12)/12 = 0.0833]
        >>> print(updated_state['X_list'][0]['close_pct'].round(4).tolist())
        [0.0, 0.0833]
        """
        if "X_list" not in state:
            return state

        transformed_X_list = []
        for df_window in state["X_list"]:
            # Apply the percentage change calculation to the current window
            if isinstance(df_window, pd.DataFrame):
                new_df = self._calculate_pct_changes(df_window)
            else:
                new_df = df_window
            transformed_X_list.append(new_df)
        
        # Update the state with the new list of transformed windows
        state["X_list"] = transformed_X_list

        return state