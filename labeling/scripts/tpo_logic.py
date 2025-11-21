import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

# --- Mocks to make your code run standalone ---
class MockAddOn:
    def add_trace_print(self, info, msg):
        # Simple print to console for debugging
        print(msg)

# --- Your Exact Logic (Unchanged internal logic) ---
def _calculate_tpo_profile(
    addon_instance: Any,
    X_df: pd.DataFrame,
    ohlc_cols: Tuple[str, ...],
    value_area_percentage: float,
    pipeline_extra_info: Dict[str, Any],
    sample_index: int
) -> Optional[pd.DataFrame]:
    
    log_prefix = f"[Sample {sample_index}] TPO Helper: "
    
    # 1. Extract all OHLC prices
    all_prices_list = []
    for col in ohlc_cols:
        all_prices_list.append(X_df[col])
        
    all_prices = pd.concat(all_prices_list).unique()
    unique_prices = np.sort(all_prices[np.isfinite(all_prices)])
    
    if len(unique_prices) < 2:
        addon_instance.add_trace_print(pipeline_extra_info, f"{log_prefix}Skipped: Not enough unique price points.")
        return None
        
    # 3. Create buckets
    buckets = []
    for i in range(len(unique_prices) - 1):
        buckets.append({
            'start': unique_prices[i],
            'end': unique_prices[i + 1],
            'volume': 0
        })
        
    if not buckets:
        return None

    # 4. Count hits
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
            
    # Mark POC
    if pocBucket:
        for bucket in buckets:
            bucket['isPoc'] = 1 if (bucket == pocBucket) else 0
    else:
        for bucket in buckets:
            bucket['isPoc'] = 0
            
    profile_df = pd.DataFrame(buckets)
    # Reverse for display
    profile_df = profile_df.iloc[::-1].reset_index(drop=True)
            
    # --- VAH/VAL Calculation ---
    total_volume = profile_df['volume'].sum()
    profile_df['isVah'] = 0
    profile_df['isVal'] = 0

    if total_volume == 0:
        return profile_df

    target_volume = total_volume * value_area_percentage
    poc_row_series = profile_df[profile_df['isPoc'] == 1]
    
    if poc_row_series.empty:
         return profile_df

    poc_index = poc_row_series.index[0]
    poc_volume = poc_row_series['volume'].iloc[0]
    
    in_value_area = pd.Series([False] * len(profile_df), index=profile_df.index)
    current_va_volume = poc_volume
    in_value_area[poc_index] = True
    
    above_ptr = poc_index - 1 
    below_ptr = poc_index + 1
    n_rows = len(profile_df)

    while current_va_volume < target_volume:
        vol_above_1 = profile_df.loc[above_ptr, 'volume'] if above_ptr >= 0 else 0
        vol_above_2 = profile_df.loc[above_ptr - 1, 'volume'] if (above_ptr - 1) >= 0 else 0
        chunk_above = vol_above_1 + vol_above_2
        
        vol_below_1 = profile_df.loc[below_ptr, 'volume'] if below_ptr < n_rows else 0
        vol_below_2 = profile_df.loc[below_ptr + 1, 'volume'] if (below_ptr + 1) < n_rows else 0
        chunk_below = vol_below_1 + vol_below_2
        
        if chunk_above == 0 and chunk_below == 0:
            break
            
        if chunk_above > chunk_below:
            current_va_volume += chunk_above
            if above_ptr >= 0: in_value_area[above_ptr] = True
            if (above_ptr - 1) >= 0: in_value_area[above_ptr - 1] = True
            above_ptr -= 2
        else:
            current_va_volume += chunk_below
            if below_ptr < n_rows: in_value_area[below_ptr] = True
            if (below_ptr + 1) < n_rows: in_value_area[below_ptr + 1] = True
            below_ptr += 2

    va_rows = profile_df[in_value_area]
    
    if not va_rows.empty:
        vah_index = va_rows.index.min()
        val_index = va_rows.index.max()
        profile_df.loc[vah_index, 'isVah'] = 1
        profile_df.loc[val_index, 'isVal'] = 1
        
    return profile_df

# --- Wrapper to Interface with Flask ---
def get_tpo_levels(df_window: pd.DataFrame):
    """
    Runs the user's logic and extracts simple float prices for POC, VAH, VAL.
    """
    mock_addon = MockAddOn()
    # Run logic
    result_df = _calculate_tpo_profile(
        addon_instance=mock_addon,
        X_df=df_window,
        ohlc_cols=('open', 'high', 'low', 'close'),
        value_area_percentage=0.70,
        pipeline_extra_info={},
        sample_index=0
    )

    if result_df is None or result_df.empty:
        return None

    # Extract Prices
    try:
        poc_row = result_df[result_df['isPoc'] == 1]
        vah_row = result_df[result_df['isVah'] == 1]
        val_row = result_df[result_df['isVal'] == 1]

        # We use the 'start' of the bucket as the price level line
        poc_price = poc_row.iloc[0]['start'] if not poc_row.empty else None
        vah_price = vah_row.iloc[0]['start'] if not vah_row.empty else None
        val_price = val_row.iloc[0]['start'] if not val_row.empty else None
        
        return {
            "POC": float(poc_price) if poc_price else 0,
            "VAH": float(vah_price) if vah_price else 0,
            "VAL": float(val_price) if val_price else 0
        }
    except Exception as e:
        print(f"Error extracting levels: {e}")
        return None