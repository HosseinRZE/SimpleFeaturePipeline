from typing import Dict, Any, Tuple, Optional
from add_ons.base_addon import BaseAddOn
import pandas as pd
import numpy as np
from data_structure.sequence_collection import SequenceCollection

def _calculate_tpo_profile(
    addon_instance: BaseAddOn,
    X_df: pd.DataFrame,
    ohlc_cols: Tuple[str, ...],
    value_area_percentage: float,  # <-- New parameter
    pipeline_extra_info: Dict[str, Any],
    sample_index: int
) -> Optional[pd.DataFrame]:
    """
    Calculates a TPO-style (hitmap) profile for a given window DataFrame.
    
    This version creates dynamic price buckets based on unique OHLC values,
    marks the Point of Control (POC), and calculates the Value Area (VAH/VAL).
    """
    log_prefix = f"[Sample {sample_index}] TPO Helper: "
    
    # 1. Extract all OHLC prices
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
            bucket['isPoc'] = 1 if (bucket == pocBucket) else 0
    else:
        for bucket in buckets:
            bucket['isPoc'] = 0
            
    # Convert to DataFrame
    profile_df = pd.DataFrame(buckets)
    
    # Reverse for display (high to low), matching your Flask app
    # After this, index 0 is the HIGHEST price.
    profile_df = profile_df.iloc[::-1].reset_index(drop=True)
            
    # --- VAH/VAL Calculation (New Logic) ---
    
    # 1. Determine total volume
    total_volume = profile_df['volume'].sum()
    
    # Initialize VAH/VAL columns
    profile_df['isVah'] = 0
    profile_df['isVal'] = 0

    if total_volume == 0:
        addon_instance.add_trace_print(
            pipeline_extra_info, 
            f"{log_prefix}Skipped VAH/VAL: Total volume is zero."
        )
        return profile_df # Return profile with 0s

    # 2. Determine target volume
    target_volume = total_volume * value_area_percentage

    # 3. Find POC row
    poc_row_series = profile_df[profile_df['isPoc'] == 1]
    
    if poc_row_series.empty:
         addon_instance.add_trace_print(
            pipeline_extra_info, 
            f"{log_prefix}Skipped VAH/VAL: POC row not found."
        )
         return profile_df # Should not happen if maxVolume > 0, but safe

    poc_index = poc_row_series.index[0]
    poc_volume = poc_row_series['volume'].iloc[0]
    
    # Track which rows are in the value area
    in_value_area = pd.Series([False] * len(profile_df), index=profile_df.index)
    
    # Add POC to Value Area
    current_va_volume = poc_volume
    in_value_area[poc_index] = True
    
    # Pointers for the *next* 2-row chunk to be evaluated
    # Remember: lower index = higher price (above)
    above_ptr = poc_index - 1 
    below_ptr = poc_index + 1
    
    n_rows = len(profile_df)

    # 4-7. Expand from POC
    while current_va_volume < target_volume:
        # Get volume of the *next* two rows "above" (lower indices)
        vol_above_1 = profile_df.loc[above_ptr, 'volume'] if above_ptr >= 0 else 0
        vol_above_2 = profile_df.loc[above_ptr - 1, 'volume'] if (above_ptr - 1) >= 0 else 0
        chunk_above = vol_above_1 + vol_above_2
        
        # Get volume of the *next* two rows "below" (higher indices)
        vol_below_1 = profile_df.loc[below_ptr, 'volume'] if below_ptr < n_rows else 0
        vol_below_2 = profile_df.loc[below_ptr + 1, 'volume'] if (below_ptr + 1) < n_rows else 0
        chunk_below = vol_below_1 + vol_below_2
        
        # If no more volume to add, stop
        if chunk_above == 0 and chunk_below == 0:
            break
            
        # Add the larger chunk
        if chunk_above > chunk_below:
            current_va_volume += chunk_above
            # Mark these rows as 'in'
            if above_ptr >= 0:
                in_value_area[above_ptr] = True
            if (above_ptr - 1) >= 0:
                in_value_area[above_ptr - 1] = True
            # Move the "above" pointer for the *next* iteration
            above_ptr -= 2
        else:
            current_va_volume += chunk_below
            # Mark these rows as 'in'
            if below_ptr < n_rows:
                in_value_area[below_ptr] = True
            if (below_ptr + 1) < n_rows:
                in_value_area[below_ptr + 1] = True
            # Move the "below" pointer for the *next* iteration
            below_ptr += 2

    addon_instance.add_trace_print(
        pipeline_extra_info, 
        f"{log_prefix}Value Area calculated ({value_area_percentage*100}%): "
        f"{current_va_volume} / {target_volume} (target) volume."
    )

    # 8. Find VAH and VAL from the collected rows
    va_rows = profile_df[in_value_area]
    
    if not va_rows.empty:
        # VAH is the highest price, which is the *lowest index*
        vah_index = va_rows.index.min()
        
        # VAL is the lowest price, which is the *highest index*
        val_index = va_rows.index.max()
        
        addon_instance.add_trace_print(
            pipeline_extra_info, 
            f"{log_prefix}VAH Index: {vah_index} (Price: {profile_df.loc[vah_index, 'start']}), "
            f"VAL Index: {val_index} (Price: {profile_df.loc[val_index, 'start']})"
        )

        # Set the 1/0 flags
        profile_df.loc[vah_index, 'isVah'] = 1
        profile_df.loc[val_index, 'isVal'] = 1
        
    # --- End of VAH/VAL Calculation ---
            
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

    The resulting profile (a DataFrame, with isPoc, isVah, isVal as 1/0) 
    is stored in `sample.X` under the key specified by `output_key`.
    """
    on_evaluation_priority = 20  # Runs after normalization (if that's at 10)

    def __init__(
        self,
        feature_group_key: str = "main",
        output_key: str = "tpo_profile",
        ohlc_cols: tuple = ("open", "high", "low", "close"),
        value_area_percentage: float = 0.70,  # <-- Configurable VA
    ):
        """
        Args:
            feature_group_key: The key in `sample.X` holding the input
                               OHLCV DataFrame for the window.
            output_key: The new key in `sample.X` where the resulting
                        TPO profile DataFrame will be stored.
            ohlc_cols: A tuple of column names for Open, High, Low, and Close.
            value_area_percentage: The percentage (e.g., 0.70 for 70%) of
                                   volume to include in the Value Area.
        """
        super().__init__()
        self.feature_group_key = feature_group_key
        self.output_key = output_key
        self.ohlc_cols = ohlc_cols
        self.value_area_percentage = value_area_percentage  # <-- Store it
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
                value_area_percentage=self.value_area_percentage, # <-- Pass it
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