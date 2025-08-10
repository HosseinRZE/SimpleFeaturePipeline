import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def compute_candle_features(df):
    """
    Assumes df has columns: 'open', 'high', 'low', 'close'
    Adds 6 extra candle features.
    """
    df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
    df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
    df['body'] = (df['close'] - df['open']).abs()

    body_range = df['body'].replace(0, 1e-6)  # avoid division by zero
    df['upper_body_ratio'] = df['upper_shadow'] / body_range
    df['lower_body_ratio'] = df['lower_shadow'] / body_range
    df['Candle_Color'] = np.where(df['close'] > df['open'], 1, 0)

    return df

def create_supervised(df, n_candles=3):
    """
    Converts sequence of candles into X, y for supervised learning.
    Keeps only rows where label is not NaN.
    """
    feature_cols = ['open','high','low','close',
                    'upper_shadow','body','lower_shadow',
                    'upper_body_ratio','lower_body_ratio','Candle_Color']

    X, y = [], []
    for i in range(n_candles, len(df)):
        if pd.notna(df.loc[i, 'label']):  # only if this row has a label
            past_candles = df.loc[i-n_candles:i-1, feature_cols].values.flatten()
            X.append(past_candles)
            y.append(df.loc[i, 'label'])

    return np.array(X), np.array(y)

def preprocess_csv(csv_path, n_candles=3):
    """
    Full preprocessing pipeline: load, feature engineer, supervised, label encode.
    """
    df = pd.read_csv(csv_path)
    
    # If only timestamp, close, label → fill dummy OHLC for now
    if not all(col in df.columns for col in ['open','high','low','close']):
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
    
    # Compute candle features
    df = compute_candle_features(df)

    # Create supervised dataset
    X, y = create_supervised(df, n_candles)

    # Encode labels to integers (0..N-1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

if __name__ == "__main__":
    X, y, le = preprocess_csv("labeled_data.csv")
    print("X shape:", X.shape)
    print("y classes:", le.classes_)
