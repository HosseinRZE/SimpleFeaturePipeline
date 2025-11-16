import pandas as pd
import numpy as np
from typing import Dict, Any, List

from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample

class PriceRateChange(BaseAddOn):
    """
    Calculates simple price ratios (Current / Previous) using the FULL data 
    (state['df_data']) to prevent per-window NaNs, and then slices the results 
    into each sequence sample.
    """
    
    def __init__(
        self, 
        feature_group_key: str = "main",
        seperatable: str = "complete", # Default to creating a new group
        dict_name: str = "price_ratios",
        relative_to: str = "same",  
        features_to_calculate: List[str] = ["open", "high", "low", "close"]
    ):
        """
        Args:
            feature_group_key: The existing feature group (dict key in sample.X) 
                               to read from and/or write to.
            seperatable: "no" -> add to feature_group_key only (merged)
                         "complete" (default) -> add to dict_name only (new feature group)
                         "both" -> add to both
            dict_name: Name for the new feature group if seperatable != "no".
            relative_to: 'same' (ratio to self previous) or 
                         'close' (ratio to previous close).
            features_to_calculate: List of columns to apply ratio to.
        """
        super().__init__()
        self.feature_group_key = feature_group_key
        self.seperatable = seperatable
        self.dict_name = dict_name
        self.relative_to = relative_to.lower()
        self.features_to_calculate = features_to_calculate

        if self.seperatable not in ("no", "complete", "both"):
            raise ValueError(f"Invalid seperatable value: {seperatable}. Must be 'no', 'complete', or 'both'.")
        if self.relative_to not in ["same", "close"]:
            raise ValueError("relative_to must be either 'same' or 'close'.")
        
        trace_msg = (
            f"PriceRateChange configured: Group='{self.feature_group_key}', Sep='{self.seperatable}', "
            f"Dict='{self.dict_name}', Relative='{self.relative_to}'"
        )
        self.add_trace_print(None, trace_msg)


    def _calculate_ratios_globally(self, full_df: pd.DataFrame, pipeline_extra_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Applies the core simple ratio calculation logic to the full DataFrame.
        """
        
        required_cols = self.features_to_calculate.copy()
        if self.relative_to == "close" and "close" not in full_df.columns:
            # Check if 'close' is even available in the full dataset
             raise ValueError("Relative to 'close' mode requires 'close' column in df_data.")

        missing_cols = [col for col in required_cols if col not in full_df.columns]
        if missing_cols:
            raise ValueError(f"Source DataFrame is missing required columns: {missing_cols}")

        global_ratio_df = pd.DataFrame(index=full_df.index)
        
        if self.relative_to == "same":
            # Ratio relative to the same feature's previous value
            for col in self.features_to_calculate:
                # Calculation: Current / Previous
                global_ratio_df[f"{col}_ratio"] = full_df[col] / full_df[col].shift(1)
        
        elif self.relative_to == "close":
            # Ratio relative to the previous close price
            prev_close = full_df["close"].shift(1)
            
            for col in self.features_to_calculate:
                # Formula: Current_Feature / Previous_Close
                global_ratio_df[f"{col}_ratio"] = full_df[col] / prev_close

        # Fill first row NaNs (due to global shift) and any Inf/-Inf (due to division by zero) with 1.0 
        # The fillna(1.0) only affects the very first row of the entire dataset.
        result_df = global_ratio_df.replace([np.inf, -np.inf], 1.0).fillna(1.0).astype(np.float32)

        # --- DEBUG: Show results of global calculation ---
        first_col = result_df.columns[0]
        print(f"[DEBUG: GLOBAL PRICE RATIO] Calculated '{first_col}'. NaNs after fill: {result_df[first_col].isnull().sum()}. First 3 rows:\n{result_df.head(3).to_string()}")
        self.add_trace_print(pipeline_extra_info, "Global Price Ratios calculated on full df_data.")
        
        return result_df


    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calculation during the training pipeline (Fit & Transform)."""
        
        self.add_trace_print(
            pipeline_extra_info, 
            f"Starting 'apply_window' for PriceRatios (Group='{self.feature_group_key}', Sep='{self.seperatable}')."
        )
        
        # 1. Get the full DataFrame
        full_df: pd.DataFrame = state.get('df_data')
        
        if full_df is None or not isinstance(full_df, pd.DataFrame):
            self.add_trace_print(pipeline_extra_info, "Skipped: state['df_data'] (Full DataFrame) is missing or not a DataFrame.")
            return state

        # 2. Perform the global calculation
        try:
            global_ratio_df = self._calculate_ratios_globally(full_df, pipeline_extra_info)
        except ValueError as e:
            self.add_trace_print(pipeline_extra_info, f"🔥 Error during global ratio calculation: {e}")
            return state


        # 3. Apply Sliced Results to each sample window
        samples: SequenceCollection = state.get("samples")
        if not samples:
            self.add_trace_print(pipeline_extra_info, "No samples found; skipping PriceRatio slicing.")
            return state

        for i, sample in enumerate(samples):
            log_prefix = f"[Sample {getattr(sample, 'original_index', i)}] "
            
            source_data = sample.X.get(self.feature_group_key)
            
            if not isinstance(source_data, pd.DataFrame):
                 self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped: Source group '{self.feature_group_key}' not a DataFrame in sample.X."
                )
                 continue

            # Get the index of the current window/sample
            window_index = source_data.index
            
            # Slice the global log returns using the window's index
            sliced_ratio_df = global_ratio_df.loc[window_index]
            
            # --- 4. Apply Seperatable Logic ---
            
            # A. Add to main feature group
            if self.seperatable in ("no", "both"):
                # Use the original source data for concatenation, not the modified one from a previous iteration
                sample.X[self.feature_group_key] = pd.concat([source_data, sliced_ratio_df], axis=1)
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Merged new features into primary group '{self.feature_group_key}'."
                )

            # B. Add to new separate feature group
            if self.seperatable in ("complete", "both"):
                sample.X[self.dict_name] = sliced_ratio_df.copy()
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Created/updated separate feature group '{self.dict_name}'."
                )

        self.add_trace_print(pipeline_extra_info, "Finished 'apply_window' for PriceRatios.")
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calculation during server inference (Transform only)."""
        
        self.add_trace_print(pipeline_extra_info, "Executing 'on_server_request' (Inference). Delegating to apply_window.")
        
        return self.apply_window(state, pipeline_extra_info)