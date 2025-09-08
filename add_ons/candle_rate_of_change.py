import numpy as np
import pandas as pd

def add_candle_ratios(
    df,
    ohlc_cols=("open", "high", "low", "close"),
    suffix="_ratio",
    relative_to="same",  # "same" = prev same column, "close" = prev close
):
    """
    Compute candle rate-of-change ratios for OHLC columns.

    Modes:
    -------
    - Ratio (same):   df[c] / prev(df[c])
    - Ratio (close):  df[c] / prev(df["close"])

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with at least columns in `ohlc_cols`.
    ohlc_cols : tuple
        Columns to compute ratios for.
    suffix : str
        Suffix for new feature columns.
    relative_to : {"same","close"}
        - "same": ratio against previous value of the same column.
        - "close": ratio against previous close.

    Returns
    -------
    pd.DataFrame
        Copy of input df with added *_ratio columns.
    """
    df = df.copy().reset_index(drop=True)

    if df.empty:
        for c in ohlc_cols:
            df[f"{c}{suffix}"] = 1.0
        return df

    for c in ohlc_cols:
        prev_same = df[c].shift(1)
        prev_close = df["close"].shift(1)

        if relative_to == "close":
            denom = prev_close.replace(0, np.nan)
            ratio = df[c] / denom
        else:  # relative_to == "same"
            denom = prev_same.replace(0, np.nan)
            ratio = df[c] / denom

        # Fill first row with 1.0 (neutral ratio)
        df[f"{c}{suffix}"] = ratio.fillna(1.0).astype(np.float32)

    return df
