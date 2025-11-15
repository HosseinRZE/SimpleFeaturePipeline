from typing import Dict, Any, Tuple, Optional
from add_ons.base_addon import BaseAddOn
import pandas as pd
import numpy as np
from data_structure.sequence_collection import SequenceCollection

# --- HELPER FUNCTION (Modified) ---

def _calculate_fixed_tpo_profile(
    addon_instance: BaseAddOn,
    X_df: pd.DataFrame,
    ohlc_cols: Tuple[str, ...],
    num_buckets: int,  # New parameter
    pipeline_extra_info: Dict[str, Any],
    sample_index: int
) -> Optional[pd.DataFrame]:
    """
    Calculates a TPO-style (hitmap) profile for a given window DataFrame
    using a *fixed number of equally spaced buckets*.

    This logic will:
    1. Find the min(low) and max(high) for the *entire* window.
    2. Divide this total price range into 'num_buckets' equal buckets.
    3. Count how many candles (rows) in the window "touch" each bucket.
    4. Identify the Point of Control (POC) - the bucket with the most hits.
    """
    log_prefix = f"[Sample {sample_index}] FixedTPO Helper: "
    
    # 1. Find the Window's Price Range
    all_prices_list = []
    for col in ohlc_cols:
        all_prices_list.append(X_df[col])
        
    all_prices = pd.concat(all_prices_list)
    window_min_price = all_prices.min()
    window_max_price = all_prices.max()
    
    # Validate the window range
    if not np.isfinite(window_min_price) or \
       not np.isfinite(window_max_price) or \
       window_min_price == window_max_price:
        addon_instance.add_trace_print(
            pipeline_extra_info, 
            f"{log_prefix}Skipped: Invalid or zero-width price range ({window_min_price} to {window_max_price})."
        )
        return None

    addon_instance.add_trace_print(
        pipeline_extra_info, 
        f"{log_prefix}Window Range: {window_min_price:.4f} to {window_max_price:.4f}. Creating {num_buckets} buckets."
    )
        
    # 2. Create the Fixed Buckets
    price_levels = np.linspace(window_min_price, window_max_price, num_buckets + 1)
    
    buckets = []
    for i in range(num_buckets):
        buckets.append({
            'start': price_levels[i],
            'end': price_levels[i + 1],
            'volume': 0  # This will be the "hit count"
        })
        
    if not buckets:
        addon_instance.add_trace_print(
            pipeline_extra_info, 
            f"{log_prefix}Skipped: No buckets were created."
        )
        return None

    # 3. Count how many times each bucket is touched
    maxVolume = 0
    pocBucket = None
    
    candle_lows = X_df[list(ohlc_cols)].min(axis=1)
    candle_highs = X_df[list(ohlc_cols)].max(axis=1)

    for bucket in buckets:
        bucket_start = bucket['start']
        bucket_end = bucket['end']
        
        overlaps = (candle_lows <= bucket_end) & (candle_highs >= bucket_start)
        hit_count = overlaps.sum()
        bucket['volume'] = hit_count

        if hit_count > maxVolume:
            maxVolume = hit_count
            pocBucket = bucket
            
    addon_instance.add_trace_print(
        pipeline_extra_info, 
        f"{log_prefix}POC found with {maxVolume} hits."
    )

    # 4. Mark POC bucket (MODIFIED: Use 1/0 instead of True/False)
    if pocBucket:
        for bucket in buckets:
            # Convert boolean (True/False) to integer (1/0)
            bucket['isPoc'] = 1 if (bucket == pocBucket) else 0
    else:
        for bucket in buckets:
            bucket['isPoc'] = 0
            
    # Convert to DataFrame
    profile_df = pd.DataFrame(buckets)
    
    # Reverse for display (high to low)
    profile_df = profile_df.iloc[::-1].reset_index(drop=True)
            
    addon_instance.add_trace_print(
        pipeline_extra_info, 
        f"{log_prefix}Profile calculation complete. Returning DataFrame with {len(profile_df)} rows."
    )
    
    return profile_df


# --- ADD-ON CLASS (Unchanged) ---

class FixedBucketTPOProfileAddOn(BaseAddOn):
    """
    Calculates a TPO-style (hitmap) volume profile for each sample window
    using a *fixed number of buckets*.

    This add-on divides the total price range of the window (min-low to
    max-high) into 'num_buckets' equally sized price buckets. It then
    counts how many candles in the window "touch" each bucket.

    The resulting profile (a DataFrame) is stored in `sample.X` under
    the key specified by `output_key`.
    """
    on_evaluation_priority = 21  # Runs after TPOProfile (if used)

    def __init__(
        self,
        num_buckets: int = 30,  # The new parameter
        feature_group_key: str = "main",
        output_key: str = "fixed_tpo_profile",
        ohlc_cols: tuple = ("open", "high", "low", "close"),
    ):
        """
        Args:
            num_buckets: The fixed number of buckets to create.
            feature_group_key: Key for the input OHLCV DataFrame.
            output_key: New key to store the resulting profile DataFrame.
            ohlc_cols: Column names for Open, High, Low, and Close.
        """
        self.num_buckets = num_buckets
        self.feature_group_key = feature_group_key
        self.output_key = output_key
        self.ohlc_cols = ohlc_cols
        self.required_cols = list(ohlc_cols)

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        
        self.add_trace_print(
            pipeline_extra_info, 
            f"Starting 'apply_window' for FixedBucketTPOProfileAddOn: Buckets={self.num_buckets}, Output='{self.output_key}'."
        )

        samples_collection: SequenceCollection = state.get("samples")
        if not samples_collection:
            self.add_trace_print(pipeline_extra_info, "Skipped: 'samples' collection is empty.")
            return state

        for i, sample in enumerate(samples_collection):
            log_prefix = f"[Sample {i}] "
            
            X_df = sample.X.get(self.feature_group_key)
            
            if X_df is None or not isinstance(X_df, pd.DataFrame):
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped: Feature group '{self.feature_group_key}' is missing or not a DataFrame."
                )
                continue
                
            missing_cols = [col for col in self.required_cols if col not in X_df.columns]
            if missing_cols:
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped: DataFrame missing columns: {missing_cols}."
                )
                continue
                
            if X_df.empty:
                self.add_trace_print(pipeline_extra_info, f"{log_prefix}Skipped: DataFrame is empty.")
                continue
            
            self.add_trace_print(
                pipeline_extra_info, 
                f"{log_prefix}Found DataFrame (Shape: {X_df.shape}). Calculating {self.num_buckets}-bucket TPO profile."
            )
            
            # Calculate the profile using the modified helper function
            profile_df = _calculate_fixed_tpo_profile(
                addon_instance=self,
                X_df=X_df,
                ohlc_cols=self.ohlc_cols,
                num_buckets=self.num_buckets, # Pass the fixed number
                pipeline_extra_info=pipeline_extra_info,
                sample_index=i
            )
            
            if profile_df is not None:
                sample.X[self.output_key] = profile_df
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Successfully saved profile to 'sample.X[{self.output_key}]'."
                )
            else:
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Fixed TPO profile calculation was skipped or returned no data."
                )

        self.add_trace_print(pipeline_extra_info, "Finished 'apply_window' for FixedBucketTPOProfileAddOn.")
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference-time hook: applies the same profile calculation.
        """
        self.add_trace_print(pipeline_extra_info, "Executing 'on_server_request' (Inference). Delegating to apply_window.")
        return self.apply_window(state, pipeline_extra_info)