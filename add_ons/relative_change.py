import pandas as pd

def add_pct_changes(
    df: pd.DataFrame,
    relative_to: str = "same",
    separatable: str = "no"  # "no", "complete", "both"
):
    """
    Add OHLC percentage change columns.

    Parameters:
        df : DataFrame with columns ["open","high","low","close"]
        relative_to : 
            - "same"  -> % change vs same field in prev candle
            - "close" -> all OHLC relative to previous close
        separatable : str
            - "no"       : default, merge features into main df only
            - "complete" : features are not merged, returned in dict only
            - "both"     : features are merged into df and also returned in dict
    """
    if separatable not in ["no", "complete", "both"]:
        raise ValueError("separatable must be 'no', 'complete', or 'both'")

    df = df.copy()

    if relative_to == "same":
        df["open_pct"] = df["open"].pct_change().fillna(0)
        df["high_pct"] = df["high"].pct_change().fillna(0)
        df["low_pct"] = df["low"].pct_change().fillna(0)
        df["close_pct"] = df["close"].pct_change().fillna(0)
    elif relative_to == "close":
        prev_close = df["close"].shift(1)
        df["open_pct"] = ((df["open"] - prev_close) / prev_close).fillna(0)
        df["high_pct"] = ((df["high"] - prev_close) / prev_close).fillna(0)
        df["low_pct"] = ((df["low"] - prev_close) / prev_close).fillna(0)
        df["close_pct"] = ((df["close"] - prev_close) / prev_close).fillna(0)
    else:
        raise ValueError("relative_to must be either 'same' or 'close'")

    # --- Handle separatable ---
    sub_df = df[["open_pct", "high_pct", "low_pct", "close_pct"]]
    if separatable == "complete":
        return df, {"pct_changes": sub_df}
    elif separatable == "both":
        return pd.concat([df, sub_df], axis=1), {"pct_changes": sub_df}
    else:  # "no"
        return df
