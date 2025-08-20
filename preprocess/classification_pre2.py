import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def create_supervised(df, n_candles=3, feature_cols=None):
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ("timestamp", "label")]

    X, y = [], []
    for i in range(n_candles, len(df)):
        if pd.notna(df.loc[i, 'label']):
            past_candles = df.loc[i-n_candles:i-1, feature_cols].values
            X.append(past_candles)
            y.append(df.loc[i, 'label'])
    return np.array(X), np.array(y)


def preprocess_csv(
    data_csv,
    labels_csv,
    n_candles=3,
    val_split=False,
    test_size=0.2,
    random_state=42,
    for_xgboost=False,
    feature_pipeline=None
):
    # Load OHLC
    df_data = pd.read_csv(data_csv)
    if not all(col in df_data.columns for col in ['open', 'high', 'low', 'close']):
        df_data['open'] = df_data['close']
        df_data['high'] = df_data['close']
        df_data['low'] = df_data['close']

    # Load labels
    df_labels = pd.read_csv(labels_csv).rename(columns={'labels': 'label'})
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
    df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'])

    # Merge
    df = pd.merge(df_data, df_labels[['timestamp', 'label']], on='timestamp', how='left')

    # 🔹 Apply feature pipeline
    if feature_pipeline is not None:
        for func in feature_pipeline:
            df = func(df)

    # 🔹 Auto-detect feature cols after pipeline
    feature_cols = [col for col in df.columns if col not in ("timestamp", "label")]

    # Create sequences
    X, y = create_supervised(df, n_candles, feature_cols)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if for_xgboost:
        X_flat = np.array([seq.flatten() for seq in X])
        if val_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y_encoded,
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded
            )
            return X_train, y_train, X_val, y_val, label_encoder, df, feature_cols
        else:
            return X_flat, y_encoded, label_encoder, df, feature_cols

    else:
        if val_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded,
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded
            )
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                        torch.tensor(y_val, dtype=torch.long))
            return train_dataset, val_dataset, label_encoder, df, feature_cols
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y_encoded, dtype=torch.long)
            dataset = TensorDataset(X_tensor, y_tensor)
            return dataset, label_encoder, df, feature_cols


def load_raw_data_serve(data_csv, labels_csv):
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


def one_sample(seq_df, feature_cols):
    """
    Given a slice of df, return (seq_len, feature_dim) array for LSTM.
    feature_cols must be the same as used during training.
    """
    return seq_df[feature_cols].values.astype(np.float32)
