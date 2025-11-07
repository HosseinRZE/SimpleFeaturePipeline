import pandas as pd
import numpy as np
from typing import Dict, Any, List

from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample

class PriceRateChange(BaseAddOn):
    """
    Calculates simple price ratios (Current / Previous) for specified features
    and adds them to the feature collection.
    
    Uses the 'seperatable' logic to determine where to store the new features.
    
    A value of 1.0 means no change.
    
    ### 💡 Example (relative_to='close'):
    
    | Time | close |
    |------|-------|
    | t-2  | 10.1  | <-- Prev Close: 10.1
    | t-1  | 10.8  | <-- Prev Close: 10.8
    | t    | 11.2  |
    
    Output 'close_ratio':
    
    | Time | close_ratio |
    |------|-------------|
    | t-2  | 1.0         | (First row)
    | t-1  | 1.0693      | (10.8 / 10.1)
    | t    | 1.0370      | (11.2 / 10.8)
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
        self.feature_group_key = feature_group_key
        self.seperatable = seperatable
        self.dict_name = dict_name
        self.relative_to = relative_to.lower()
        self.features_to_calculate = features_to_calculate

        if self.seperatable not in ("no", "complete", "both"):
            raise ValueError(f"Invalid seperatable value: {seperatable}. Must be 'no', 'complete', or 'both'.")

        if self.relative_to not in ["same", "close"]:
            raise ValueError("relative_to must be either 'same' or 'close'.")
        
        # Check for required 'close' column if in 'close' mode
        if self.relative_to == "close" and "close" not in self.features_to_calculate:
             self.add_trace_print(None, "Warning: relative_to='close' but 'close' not in features_to_calculate. Adding 'close'.")
             # Silently add 'close' as it's required for the calculation,
             # but it will only be calculated if it was *also* in the original list.
             # This check is more about *source* data.
             pass # The check in _calculate_ratios is the important one.

        trace_msg = (
            f"PriceRateChange configured: Group='{self.feature_group_key}', Sep='{self.seperatable}', "
            f"Dict='{self.dict_name}', Relative='{self.relative_to}'"
        )
        self.add_trace_print(None, trace_msg)


    def _calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the core simple ratio calculation logic to a single DataFrame.
        """
        
        # Check for source columns needed for calculation
        required_cols = self.features_to_calculate.copy()
        if self.relative_to == "close" and "close" not in required_cols:
             # 'close' is needed as a denominator, even if not a target feature
            required_cols.append("close") 

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Source group DataFrame is missing required columns: {missing_cols}")

        df_out = pd.DataFrame(index=df.index)
        
        if self.relative_to == "same":
            # Ratio relative to the same feature's previous value
            for col in self.features_to_calculate:
                # Calculation: Current / Previous
                df_out[f"{col}_ratio"] = df[col] / df[col].shift(1)
        
        elif self.relative_to == "close":
            # Ratio relative to the previous close price
            prev_close = df["close"].shift(1)
            
            for col in self.features_to_calculate:
                # Formula: Current_Feature / Previous_Close
                df_out[f"{col}_ratio"] = df[col] / prev_close

        # Fill first row NaNs (due to shift) and any Inf/-Inf (due to division by zero) with 1.0 (no change)
        return df_out.replace([np.inf, -np.inf], 1.0).fillna(1.0).astype(np.float32)

    def _process_samples(self, samples: SequenceCollection, pipeline_extra_info: Dict[str, Any]):
        """
        Iterates through samples and applies the calculation to each one.
        """
        
        for i, sample in enumerate(samples):
            log_prefix = f"[Sample {getattr(sample, 'original_index', i)}] "
            
            # 1. Get the target DataFrame
            source_data = sample.X.get(self.feature_group_key)
            
            if not isinstance(source_data, pd.DataFrame):
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped: Feature group '{self.feature_group_key}' is missing or not a DataFrame."
                )
                continue

            try:
                # 2. Compute Features
                ratio_df = self._calculate_ratios(source_data)
                
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Computed {len(ratio_df.columns)} PriceRatio features (Shape: {ratio_df.shape})."
                )

                # 3. Apply Seperatable Logic
                
                # A. Add to main feature group
                if self.seperatable in ("no", "both"):
                    sample.X[self.feature_group_key] = pd.concat([source_data, ratio_df], axis=1)
                    self.add_trace_print(
                        pipeline_extra_info, 
                        f"{log_prefix}Merged new features into primary group '{self.feature_group_key}'."
                    )

                # B. Add to new separate feature group
                if self.seperatable in ("complete", "both"):
                    sample.X[self.dict_name] = ratio_df.copy()
                    self.add_trace_print(
                        pipeline_extra_info, 
                        f"{log_prefix}Created/updated separate feature group '{self.dict_name}'."
                    )

            except Exception as e:
                self.add_trace_print(pipeline_extra_info, f"🔥 {log_prefix}Error calculating PriceRatios: {e}")

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calculation during the training pipeline (Fit & Transform)."""
        
        self.add_trace_print(
            pipeline_extra_info, 
            f"Starting 'apply_window' for PriceRatios (Group='{self.feature_group_key}', Sep='{self.seperatable}')."
        )
        
        samples: SequenceCollection = state.get("samples")
        if not samples:
            self.add_trace_print(pipeline_extra_info, "No samples found; skipping PriceRatio calculation.")
            return state

        self._process_samples(samples, pipeline_extra_info)
        
        self.add_trace_print(pipeline_extra_info, "Finished 'apply_window' for PriceRatios.")
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calculation during server inference (Transform only)."""
        
        self.add_trace_print(pipeline_extra_info, "Executing 'on_server_request' (Inference). Delegating to apply_window.")
        
        return self.apply_window(state, pipeline_extra_info)