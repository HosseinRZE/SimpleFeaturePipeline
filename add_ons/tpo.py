from typing import Dict, Any, Tuple, Optional
from add_ons.base_addon import BaseAddOn
import pandas as pd
import numpy as np
from data_structure.sequence_collection import SequenceCollection

def _calculate_tpo_profile(
    addon_instance: BaseAddOn,
    X_df: pd.DataFrame,
    ohlc_cols: Tuple[str, ...],
    pipeline_extra_info: Dict[str, Any],
    sample_index: int
) -> Optional[pd.DataFrame]:
    """
    Calculates a TPO-style (hitmap) profile for a given window DataFrame.
    
    This version creates dynamic price buckets based on unique OHLC values
    and marks the Point of Control (POC) with 1 (otherwise 0).
    """
    log_prefix = f"[Sample {sample_index}] TPO Helper: "
    
    # 1. Extract all OHLC prices from all candles in the window
    all_prices_list = []
    for col in ohlc_cols:
        all_prices_list.append(X_df[col])
        
    all_prices = pd.concat(all_prices_list).unique()
    unique_prices = np.sort(all_prices[np.isfinite(all_prices)])
    
    if len(unique_prices) < 2:
        addon_instance.add_trace_print(
            pipeline_extra_info, 
            f"{log_prefix}Skipped: Not enough unique price points ({len(unique_prices)}) to build buckets."
        )
        return None
        
    addon_instance.add_trace_print(
        pipeline_extra_info, 
        f"{log_prefix}Found {len(unique_prices)} unique price levels."
    )
        
    # 3. Create buckets between consecutive prices
    buckets = []
    for i in range(len(unique_prices) - 1):
        buckets.append({
            'start': unique_prices[i],
            'end': unique_prices[i + 1],
            'volume': 0  # This will be the "hit count"
        })
        
    if not buckets:
        addon_instance.add_trace_print(
            pipeline_extra_info, 
            f"{log_prefix}Skipped: No buckets were created."
        )
        return None

    addon_instance.add_trace_print(
        pipeline_extra_info, 
        f"{log_prefix}Created {len(buckets)} price buckets."
    )

    # 4. Count how many times each bucket is touched by candle ranges
    maxVolume = 0
    pocBucket = None
    
    # Pre-calculate candle ranges
    candle_lows = X_df[list(ohlc_cols)].min(axis=1)
    candle_highs = X_df[list(ohlc_cols)].max(axis=1)

    for bucket in buckets:
        bucket_start = bucket['start']
        bucket_end = bucket['end']
        
        # Vectorized check for overlap
        overlaps = (candle_lows <= bucket_end) & (candle_highs >= bucket_start)
        
        # Sum the booleans (True=1, False=0) to get the hit count
        hit_count = overlaps.sum()
        bucket['volume'] = hit_count

        if hit_count > maxVolume:
            maxVolume = hit_count
            pocBucket = bucket
            
    addon_instance.add_trace_print(
        pipeline_extra_info, 
        f"{log_prefix}POC found with {maxVolume} hits."
    )

    # Mark POC bucket (MODIFIED: Use 1/0 instead of True/False)
    if pocBucket:
        for bucket in buckets:
            # Convert boolean (True/False) to integer (1/0)
            bucket['isPoc'] = 1 if (bucket == pocBucket) else 0
    else:
        for bucket in buckets:
            bucket['isPoc'] = 0
            
    # Convert to DataFrame
    profile_df = pd.DataFrame(buckets)
    
    # Reverse for display (high to low), matching your Flask app
    profile_df = profile_df.iloc[::-1].reset_index(drop=True)
            
    addon_instance.add_trace_print(
        pipeline_extra_info, 
        f"{log_prefix}Profile calculation complete. Returning DataFrame with {len(profile_df)} rows."
    )
    
    return profile_df


class TPOProfileAddOn(BaseAddOn):
    """
    Calculates a TPO-style (hitmap) volume profile for each sample window.

    This add-on mirrors the logic from the user's Flask /calculate_profile
    endpoint. It does NOT use the 'volume' column, but instead counts
    how many candles in the window touch each discrete price level.

    The resulting profile (a DataFrame, with isPoc as 1/0) is stored in 
    `sample.X` under the key specified by `output_key`.
    """
    on_evaluation_priority = 20  # Runs after normalization (if that's at 10)

    def __init__(
        self,
        feature_group_key: str = "main",
        output_key: str = "tpo_profile",
        ohlc_cols: tuple = ("open", "high", "low", "close"),
    ):
        """
        Args:
            feature_group_key: The key in `sample.X` holding the input
                               OHLCV DataFrame for the window.
            output_key: The new key in `sample.X` where the resulting
                        TPO profile DataFrame will be stored.
            ohlc_cols: A tuple of column names for Open, High, Low, and Close.
        """
        super().__init__()
        self.feature_group_key = feature_group_key
        self.output_key = output_key
        self.ohlc_cols = ohlc_cols
        self.required_cols = list(ohlc_cols)

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        
        self.add_trace_print(
            pipeline_extra_info, 
            f"Starting 'apply_window' for TPOProfileAddOn: Input='{self.feature_group_key}', Output='{self.output_key}'."
        )

        samples_collection: SequenceCollection = state.get("samples")
        if not samples_collection:
            self.add_trace_print(pipeline_extra_info, "Skipped: 'samples' collection is empty or missing in state.")
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
                
            # Check for required columns
            missing_cols = [col for col in self.required_cols if col not in X_df.columns]
            if missing_cols:
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped: DataFrame is missing required columns: {missing_cols}."
                )
                continue
                
            if X_df.empty:
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped: DataFrame is empty."
                )
                continue
            
            self.add_trace_print(
                pipeline_extra_info, 
                f"{log_prefix}Found DataFrame (Shape: {X_df.shape}). Calculating TPO profile."
            )
            
            # Calculate the profile using the helper function
            profile_df = _calculate_tpo_profile(
                addon_instance=self,
                X_df=X_df,
                ohlc_cols=self.ohlc_cols,
                pipeline_extra_info=pipeline_extra_info,
                sample_index=i
            )
            
            if profile_df is not None:
                # Store the resulting profile DataFrame in the sample
                sample.X[self.output_key] = profile_df
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Successfully saved TPO profile to 'sample.X[{self.output_key}]'."
                )
            else:
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}TPO profile calculation was skipped or returned no data."
                )

        self.add_trace_print(pipeline_extra_info, "Finished 'apply_window' for TPOProfileAddOn.")
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference-time hook: applies the same profile calculation.
        """
        self.add_trace_print(pipeline_extra_info, "Executing 'on_server_request' (Inference). Delegating to apply_window.")
        return self.apply_window(state, pipeline_extra_info)