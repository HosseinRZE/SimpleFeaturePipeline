import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    'open', 'high', 'low', 'close',
    'upper_shadow', 'body', 'lower_shadow',
    'upper_body_ratio', 'lower_body_ratio', 'Candle_Color'
]

def create_supervised(df, n_candles=3):
    X, y = [], []
    for i in range(n_candles, len(df)):
        if pd.notna(df.loc[i, 'label']):  # only if label exists
            past_candles = df.loc[i-n_candles:i-1, FEATURE_COLS].values
            X.append(past_candles)  # shape: (n_candles, feature_dim)
            y.append(df.loc[i, 'label'])
    return np.array(X), np.array(y)


def preprocess_csv(data_csv, labels_csv, n_candles=3, val_split=False, test_size=0.2, random_state=42):
    """
    Preprocesses when OHLC data and labels are in separate CSVs.
    If val_split=True, returns (train_dataset, val_dataset, label_encoder, merged_df).
    Otherwise returns (dataset, label_encoder, merged_df).
    """
    # Load OHLC
    df_data = pd.read_csv(data_csv)
    if not all(col in df_data.columns for col in ['open', 'high', 'low', 'close']):
        df_data['open'] = df_data['close']
        df_data['high'] = df_data['close']
        df_data['low'] = df_data['close']

    # Load labels
    df_labels = pd.read_csv(labels_csv).rename(columns={'labels': 'label'})

    # Align timestamps
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
    df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'])

    # Merge
    df = pd.merge(df_data, df_labels[['timestamp', 'label']], on='timestamp', how='left')

    # Supervised dataset
    X, y = create_supervised(df, n_candles)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if val_split:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.long))
        return train_dataset, val_dataset, label_encoder, df
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        return dataset, label_encoder, df

def load_raw_data_serve(data_csv, labels_csv):
    """
    Server version — only loads and merges, does not encode or sequence.
    """
    df_data = pd.read_csv(data_csv)
    if not all(col in df_data.columns for col in ['open', 'high', 'low', 'close']):
        df_data['open'] = df_data['close']
        df_data['high'] = df_data['close']
        df_data['low'] = df_data['close']

    df_labels = pd.read_csv(labels_csv).rename(columns={'labels': 'label'})
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
    df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'])

    df = pd.merge(df_data, df_labels[['timestamp', 'label']], on='timestamp', how='left')
    return df

def one_sample(seq_df):
    """
    Given a slice of df, return (seq_len, feature_dim) array for LSTM.
    """
    return seq_df[FEATURE_COLS].values.astype(np.float32)
