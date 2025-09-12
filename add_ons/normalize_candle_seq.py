import numpy as np
import pandas as pd

def add_label_normalized_candles(
    df,
    ohlc_cols=("open","high","low","close"),
    suffix="_prop",
):
    """
    For a single subsequence (window), compute per-candle proportions:
    each candle's OHLC divided by the window's *last* close.

    Adds 4 columns to df: open_prop, high_prop, low_prop, close_prop (float32).
    """
    df = df.copy().reset_index(drop=True)

    if df.empty:
        for c in ohlc_cols:
            df[f"{c}{suffix}"] = 0.0
        return df

    last_close = df["close"].iloc[-1]
    if not np.isfinite(last_close) or last_close == 0.0:
        # avoid NaNs or divide-by-zero
        for c in ohlc_cols:
            df[f"{c}{suffix}"] = 0.0
        return df

    for c in ohlc_cols:
        df[f"{c}{suffix}"] = df[c].astype(np.float32) / last_close

    return df, []
