import numpy as np
import pandas as pd

SEQ_LEN   = 3            # ← exactly 3 consecutive days
FEATURES  = ['open', 'high', 'low', 'close', 'upper_shadow',
             'body', 'lower_shadow', 'upper_body_ratio',
             'lower_body_ratio', 'Candle_Color']   # 10-D per timestep

def load_raw_data(candle_csv, label_csv):
    candles = pd.read_csv(candle_csv, parse_dates=['timestamp'])
    labels  = pd.read_csv(label_csv, header=None,
                          names=['timestamp', 'last_close', 'line_raw'])
    labels['timestamp'] = pd.to_datetime(labels['timestamp'])

    # 1. left merge keeps every candle row
    df = pd.merge(candles, labels[['timestamp', 'line_raw']], on='timestamp', how='left')
    df = df.set_index('timestamp').asfreq('D').ffill()   # daily, fill OHLC

    # 2. label: 1 if price exists, 0 otherwise
    df['label_price'] = df['line_raw'].str.extract(r"\(([^)]+)\)")[0].astype(float)
    df['has_line']    = (df['label_price'] != -1) & df['label_price'].notna()
    df['log_factor']  = np.where(df['has_line'],
                                 np.log(np.maximum(df['label_price'], 1e-6)),
                                 np.nan)

    return df                                           # **all** calendar days

def build_sequences(df):
    X_seq, y_cls, y_reg = [], [], []
    for i in range(SEQ_LEN, len(df)):
        seq = df.iloc[i - SEQ_LEN : i][FEATURES].values.astype(np.float32)

        cls = float(df.iloc[i]['has_line'])
        reg = float(df.iloc[i]['log_factor'])


        X_seq.append(seq)
        y_cls.append(cls)
        y_reg.append(reg)

    return np.array(X_seq), np.array(y_cls), np.array(y_reg)

def one_sample(df_slice):
    """
    df_slice : DataFrame with exactly seq_len rows
    returns  : ndarray shape (seq_len, feature_dim)
    """
    return df_slice[FEATURES].values.astype(np.float32)

from pandas.tseries.offsets import Day

def load_raw_data_serve(candle_csv, label_csv):
    candles = pd.read_csv(candle_csv, parse_dates=['timestamp'])
    labels  = pd.read_csv(label_csv, header=None,
                          names=['timestamp', 'last_close', 'line_raw'])
    labels['timestamp'] = pd.to_datetime(labels['timestamp'])

    # --- FIX 1: Change 'inner' to 'left' ---
    # This keeps ALL rows from the 'candles' dataframe.
    df = pd.merge(candles, labels[['timestamp', 'line_raw']], on='timestamp', how='left')

    # The rest of the function can now handle the possibility of missing labels
    # The .get() method is safer than direct access ['line_raw'] if a column might not exist
    # after a failed merge, but here we can rely on `how=left` to create the column.
    
    # We need to handle potential NaN values from the left merge
    df['label_price'] = df['line_raw'].str.extract(r"\(([^)]+)\)")[0].astype(float)
    df['has_line']    = (df['label_price'] != -1) & (df['label_price'].notna())
    df['has_line']    = df['has_line'].astype(int)
    
    df['log_factor']  = np.where(df['has_line'],
                                  np.log(np.maximum(df['label_price'], 1e-6)),
                                  np.nan)

    # --- FIX 2: Remove the dropna call ---
    # Do NOT drop rows, as we need all candles for the chart display.
    # return df.dropna(subset=['log_factor'])  <-- COMMENT OUT OR DELETE THIS LINE

    return df # Return the full dataframe with all original candles
