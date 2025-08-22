import pandas as pd
import ta  

def add_rsi(
    df: pd.DataFrame,
    length: int = 14,
    separatable: str = "no"  # "no", "complete", "both"
):
    """
    Add RSI (Relative Strength Index) feature to DataFrame.

    Parameters:
        df : DataFrame with "close" column
        length : RSI period (default 14)
        separatable : str
            - "no"       : merge RSI into main df only
            - "complete" : return RSI in dict only
            - "both"     : merge into df and return in dict
    """
    if separatable not in ["no", "complete", "both"]:
        raise ValueError("separatable must be 'no', 'complete', or 'both'")

    df = df.copy()
    rsi_series = ta.momentum.RSIIndicator(close=df["close"], window=length).rsi().fillna(0)
    sub_df = pd.DataFrame({"rsi": rsi_series}, index=df.index)

    if separatable == "complete":
        return df, {"rsi": sub_df}
    elif separatable == "both":
        df = pd.concat([df, sub_df], axis=1)
        return df, {"rsi": sub_df}
    else:  # "no"
        df = pd.concat([df, sub_df], axis=1)
        return df
