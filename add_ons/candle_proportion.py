import numpy as np
import pandas as pd 

def add_candle_proportions(
    df,
    period=50,
    additional_props=False,
    include_candle_color=False,
    abs_props=True,
    upper_bound=5,
    return_last_row_only=False,
    init_atr=688,
    separatable="no",   # "no", "complete", "both"
):
    """
    Add ATR-normalized candlestick proportions as new features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, must contain columns ["open", "high", "low", "close"].
    period : int, default=50
        Lookback period for ATR (Average True Range) calculation using Wilder's smoothing.
    additional_props : bool, default=False
        If True, add ratio-based features (upper/body, lower/body, upper/lower).
    include_candle_color : bool, default=False
        If True, add a column "candle_color" (1=red candle, 2=green candle).
    abs_props : bool, default=True
        If True, take absolute values of ratio features.
    upper_bound : float, default=5
        Clip values of all generated features at this bound to reduce outliers.
    return_last_row_only : bool, default=False
        If True, return only the last row of features (useful for live trading inference).
    separatable : {"no", "complete", "both"}, default="no"
        - "no"       : merge features into main DataFrame only
        - "complete" : return features only (dict with key "candle_props")
        - "both"     : return merged DataFrame and features dict

    Returns
    -------
    pd.DataFrame or dict
        Depending on `separatable`:
        - "no"       : DataFrame with new features merged
        - "complete" : {"candle_props": DataFrame with only features}
        - "both"     : {"main": merged DataFrame, "candle_props": features only}

    Added Columns (formulas are normalized by ATR)
    ----------------------------------------------
    1. upper_shadow :
        - Green candle: (high - close) / ATR
        - Red candle  : (high - open) / ATR
        Represents the relative size of the wick above the candle body.

    2. body :
        (close - open) / ATR  
        Positive for green candles, negative for red candles.  
        Represents the candle body height relative to ATR.

    3. lower_shadow :
        - Green candle: (open - low) / ATR
        - Red candle  : (close - low) / ATR  
        Represents the relative size of the wick below the body.

    4. candle_color (if include_candle_color=True) :
        - 1 = red candle (open > close)
        - 2 = green candle (close >= open)

    5. upper_body_ratio (if additional_props=True) :
        upper_shadow / body  
        Ratio of upper wick size relative to body.

    6. lower_body_ratio (if additional_props=True) :
        lower_shadow / body  
        Ratio of lower wick size relative to body.

    7. upper_lower_body_ratio (if additional_props=True) :
        upper_shadow / lower_shadow  
        Balance between upper and lower wick sizes.

    Notes
    -----
    - ATR is computed with Wilder's smoothing.
    - All values are clipped to +/- `upper_bound`.
    - Infinite and NaN ratios are replaced with 0.
    - Features are designed to be scale-invariant across different assets.

    Examples
    --------
    >>> pipeline = FeaturePipeline(steps=[
    ...     lambda df: add_candle_proportions(df, separatable="no", additional_props=True)
    ... ])

    >>> pipeline = FeaturePipeline(steps=[
    ...     lambda df: add_candle_proportions(df, separatable="complete")
    ... ])

    >>> pipeline = FeaturePipeline(steps=[
    ...     lambda df: add_candle_proportions(df, separatable="both", include_candle_color=True)
    ... ])
    df_with_props = add_candle_proportions(
    df,
    period=50,
    additional_props=True,        # add ratios
    include_candle_color=True,    # add 1=red, 2=green
    separatable="no"              # merge into df
)
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    df = df.copy().reset_index(drop=True)

    # --- ATR calculation ---
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    tr.iloc[0] = (high - low).iloc[0]

    atr = np.zeros(len(tr))
    if len(tr) > 0:
        # First ATR value is averaged with provided init_atr
        atr[0] = (init_atr + tr.iloc[0]) / 2

    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr.iloc[i]) / period
    atr_series = pd.Series(atr, index=df.index)

    # --- Candle features ---
    red_cndl = df["open"] > df["close"]
    green_cndl = ~red_cndl

    upper_shadow = np.where(
        green_cndl, (df["high"] - df["close"]) / atr_series,
        (df["high"] - df["open"]) / atr_series
    )
    body = (df["close"] - df["open"]) / atr_series
    lower_shadow = np.where(
        green_cndl, (df["open"] - df["low"]) / atr_series,
        (df["close"] - df["low"]) / atr_series
    )

    out = pd.DataFrame({
        "upper_shadow": upper_shadow,
        "body": body,
        "lower_shadow": lower_shadow
    }, index=df.index)

    # Clip
    for col in ["upper_shadow", "body", "lower_shadow"]:
        out[col] = out[col].clip(upper=upper_bound)

    cols = ["upper_shadow", "body", "lower_shadow"]

    if include_candle_color:
        out["candle_color"] = np.where(red_cndl, 1, 2)
        cols.append("candle_color")

    if additional_props:
        ubr = (out["upper_shadow"] / out["body"]).replace([np.inf, -np.inf], 0).fillna(0)
        lbr = (out["lower_shadow"] / out["body"]).replace([np.inf, -np.inf], 0).fillna(0)
        ulbr = (out["upper_shadow"] / out["lower_shadow"]).replace([np.inf, -np.inf], 0).fillna(0)

        if abs_props:
            ubr, lbr, ulbr = ubr.abs(), lbr.abs(), ulbr.abs()

        out["upper_body_ratio"] = ubr.clip(upper=upper_bound)
        out["lower_body_ratio"] = lbr.clip(upper=upper_bound)
        out["upper_lower_body_ratio"] = ulbr.clip(upper=upper_bound)
        cols.extend(["upper_body_ratio", "lower_body_ratio", "upper_lower_body_ratio"])

    if return_last_row_only:
        out = out.iloc[-1:]

    # --- separatable logic ---
    if separatable == "complete":
        return {"candle_props": out[cols]}
    elif separatable == "both":
        merged = pd.concat([df, out[cols]], axis=1)
        return {"main": merged, "candle_props": out[cols]}
    else:  # "no"
        return pd.concat([df, out[cols]], axis=1)