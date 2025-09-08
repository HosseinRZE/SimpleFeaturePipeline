import numpy as np
import pandas as pd

def add_candle_rocp(
    df,
    ohlc_cols=("open", "high", "low", "close"),
    suffix="_dif",
    normalize=True,
    relative_to="same",  # "same" = prev same column, "close" = prev close
):
    """
    Compute candle differences for OHLC columns.

    Modes:
    -------
    - Raw difference:       df[c] - prev(df[c])
    - Normalized (same):    (df[c] - prev(df[c])) / prev(df[c])
    - Normalized (close):   (df[c] - prev(df["close"])) / prev(df["close"])

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with at least columns in `ohlc_cols`.
    ohlc_cols : tuple
        Columns to compute differences for.
    suffix : str
        Suffix for new feature columns.
    normalize : bool
        Whether to normalize differences by previous value.
    relative_to : {"same","close"}
        - "same": normalize by previous value of the same column.
        - "close": normalize by previous close.

    Returns
    -------
    pd.DataFrame
        Copy of input df with added *_dif columns.
    """
    df = df.copy().reset_index(drop=True)

    if df.empty:
        for c in ohlc_cols:
            df[f"{c}{suffix}"] = 0.0
        return df

    for c in ohlc_cols:
        prev_same = df[c].shift(1)
        prev_close = df["close"].shift(1)

        if normalize:
            if relative_to == "close":
                denom = prev_close.replace(0, np.nan)
                dif = (df[c] - prev_close) / denom
            else:  # relative_to == "same"
                denom = prev_same.replace(0, np.nan)
                dif = (df[c] - prev_same) / denom
        else:
            dif = df[c] - prev_same

        df[f"{c}{suffix}"] = dif.fillna(0.0).astype(np.float32)

    return df
