import pandas as pd
import numpy as np

def add_candle_sequences(
    df,
    seq_len=10,
    separatable="no",  # "no", "complete", "both"
):
    """
    Generate normalized OHLC sequences for CNN input.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ["open", "high", "low", "close"].
    seq_len : int, default=10
        Number of candles in each sequence.
    separatable : str, default="no"
        - "no"       : merge sequences into main df only
        - "complete" : return only dict { "candle_seq": sequences }
        - "both"     : return {"main": df_with_sequences, "candle_seq": sequences}

    Returns
    -------
    Depending on `separatable`:
    - "no"       : DataFrame with original columns + "candle_seq"
    - "complete" : dict { "candle_seq": np.ndarray of shape (n, seq_len, 4) }
    - "both"     : dict { "main": df_with_sequences, "candle_seq": np.ndarray }

    Notes
    -----
    - Each sequence is `seq_len` candles with features [open, high, low, close],
      normalized by the last candle's close price in that sequence.
    - Padding: for the first `seq_len-1` rows, the missing candles are
      filled by duplicating the earliest available candle until the
      sequence length is satisfied.
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    df = df.reset_index(drop=True)
    num_rows = len(df)
    if num_rows == 0:
        raise ValueError("Empty DataFrame passed.")

    sequences = []
    for i in range(num_rows):
        if i + 1 < seq_len:
            # Not enough candles, pad with earliest candle
            window = df.iloc[:i+1][["open", "high", "low", "close"]].values
            pad_len = seq_len - (i+1)
            pad_candle = np.repeat(window[0][np.newaxis, :], pad_len, axis=0)
            window = np.vstack([pad_candle, window])
        else:
            # Full sequence
            window = df.iloc[i-seq_len+1:i+1][["open", "high", "low", "close"]].values

        last_close = window[-1, 3]  # close of last candle
        norm_window = window / last_close
        sequences.append(norm_window)

    sequences = np.array(sequences)  # shape (num_rows, seq_len, 4)

    # attach to df if needed
    if separatable == "complete":
        return {"candle_seq": sequences}
    elif separatable == "both":
        df_out = df.copy()
        df_out["candle_seq"] = list(sequences)
        return {"main": df_out, "candle_seq": sequences}
    else:  # "no"
        df_out = df.copy()
        df_out["candle_seq"] = list(sequences)
        return df_out
