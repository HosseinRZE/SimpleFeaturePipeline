import pandas as pd
from utils.zigzag_bandf import ZigZag

def add_zigzag(df, window_size=3, dev_threshold=1, max_pivots=10):
    """
    Append zigzag features (swing highs/lows) to dataframe.
    """
    zz = ZigZag(window_size=window_size, dev_threshold=dev_threshold, max_pivots=max_pivots)

    df = df.copy()
    highs, lows = [], []

    for idx, row in df.iterrows():
        zz.update(idx, row["high"], row["low"], row["close"])
        high_val, low_val = None, None
        if zz.pivots:
            last_pivot = zz.pivots[-1]
            if last_pivot.is_high == True:
                high_val = last_pivot.price
            elif last_pivot.is_high == False:
                low_val = last_pivot.price
        highs.append(high_val)
        lows.append(low_val)

    df["zigzag_high"] = highs
    df["zigzag_low"] = lows

    # Fill forward so we can use these in sequences
    df["zigzag_high"].ffill(inplace=True)
    df["zigzag_low"].ffill(inplace=True)

    return df