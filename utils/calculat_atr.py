import numpy as np
import pandas as pd

def calculate_atr(df, period=50):
    """
    Calculates the Average True Range (ATR) based on the user's provided logic.
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
    init_period = min(period, len(tr))
    atr[0] = tr.iloc[:init_period].mean()
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr.iloc[i]) / period
    
    return pd.Series(atr, index=df.index)