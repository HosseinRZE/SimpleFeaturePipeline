# In your feature engineering file
import numpy as np
import pandas as pd

def add_candle_rocp(df, **kwargs):
    """
    Computes candle differences and returns a tuple: 
    (DataFrame, list_of_invalid_indices)
    """
    # Assuming the implementation from the previous answer
    df_out = df.copy()
    invalid_indices = set()

    # Your feature logic here...
    # Example: find rows where a shift operation creates NaNs
    # dif = df_out['close'].shift(1)
    # bad_rows = dif[dif.isna()].index
    # invalid_indices.update(bad_rows)
    # ...

    # At the end, return both the modified dataframe and the indices
    # return df_out, list(invalid_indices)
    
    # Using the full implementation from before:
    ohlc_cols=kwargs.get("ohlc_cols", ("open", "high", "low", "close"))
    suffix=kwargs.get("suffix", "_dif")
    normalize=kwargs.get("normalize", True)
    relative_to=kwargs.get("relative_to", "close")

    if df.empty:
        for c in ohlc_cols:
            df_out[f"{c}{suffix}"] = 0.0
        return df_out, []

    for c in ohlc_cols:
        prev_same = df_out[c].shift(1)
        prev_close = df_out["close"].shift(1)

        if normalize:
            if relative_to == "close":
                denom = prev_close.replace(0, np.nan)
                dif = (df_out[c] - prev_close) / denom
            else:
                denom = prev_same.replace(0, np.nan)
                dif = (df_out[c] - prev_same) / denom
        else:
            dif = df_out[c] - prev_same

        bad_rows = dif[dif.isna()].index
        invalid_indices.update(bad_rows)
        df_out[f"{c}{suffix}"] = dif.fillna(0.0).astype(np.float32)

    return df_out, list(invalid_indices)