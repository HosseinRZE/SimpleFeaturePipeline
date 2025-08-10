# predictor.py
import numpy as np
import pandas as pd

SEQ_LEN = 5
LENGTHS = (1, 3, 5)

def make_features(candles, lengths=(1,3,5)):
    max_len = max(lengths)
    X, y_cls, y_reg = [], [], []
    for i in range(max_len, len(candles)+1):          # inclusive upper bound
        parts = []
        for L in lengths:
            seq = candles.iloc[i-L:i]                # i-L to i-1
            norm = seq.iloc[-1]['close']
            parts.append((seq[['open','high','low','close']] / norm).values.astype(float).ravel())

        meta = candles.iloc[i-1][['upper_shadow','body','lower_shadow',
                                  'upper_body_ratio','lower_body_ratio','Candle_Color']].values.astype(float)
        X.append(np.concatenate(parts + [meta]).astype(float))
        y_cls.append(candles.iloc[i-1]['has_line'])
        y_reg.append(candles.iloc[i-1]['log_coef'])
    return np.array(X, dtype=np.float32), np.array(y_cls), np.array(y_reg)

# predictor.py  (append at bottom)
def one_feature_vector(candles_df):
    """
    candles_df : DataFrame with exactly SEQ_LEN rows
    returns      : ndarray shape (42,)
    """
    X, _, _ = make_features(candles_df)   # (1, 42)
    return X[-1]                          # (42,)

def load_data(candle_path, label_path):
    df_c = pd.read_csv(candle_path, parse_dates=['timestamp'])
    df_l = pd.read_csv(label_path, header=None,
                       names=['timestamp', 'last_close', 'line_raw'])
    df_l['timestamp'] = pd.to_datetime(df_l['timestamp'])
    df_l['line1'] = df_l['line_raw'].str.extract(r"\(([^)]+)\)")[0].astype(float)

    df = pd.merge(df_c, df_l[['timestamp', 'line1']], on='timestamp', how='inner')
    df['has_line'] = (df['line1'] != -1).astype(int)
    df['log_coef'] = np.where(df['has_line'] == 1, np.log(df['line1']), np.nan)
    return df