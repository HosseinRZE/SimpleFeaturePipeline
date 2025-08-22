import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- MultiInputDataset for dict of sequences ---
class MultiInputDataset(Dataset):
    def __init__(self, X_dict, y):
        """
        X_dict: dict of {feature_group_name: np.array of shape (num_samples, seq_len, feat_dim)}
        y: np.array of labels
        """
        self.X_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in X_dict.items()}
        self.y = torch.tensor(y, dtype=torch.long)
        self.length = len(y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.X_dict.items()}
        return sample, self.y[idx]


# --- Modified create_supervised ---
def create_supervised(df, n_candles, feature_cols=None, labels=None):
    """
    If n_candles is an int: normal behavior
    If n_candles is dict: each key = feature group, value = sequence length
    labels: Series or array of labels (used if df has no 'label')
    """
    if isinstance(n_candles, int):
        # single int for all features
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in ("timestamp", "label")]

        X, y = [], []
        for i in range(n_candles, len(df)):
            label = df.loc[i, 'label'] if 'label' in df.columns else labels[i]
            if pd.notna(label):
                past_candles = df.loc[i-n_candles:i-1, feature_cols].values
                X.append(past_candles)
                y.append(label)
        return np.array(X), np.array(y)

    elif isinstance(n_candles, dict):
        raise ValueError("Dict mode must be handled outside, per feature group.")

    else:
        raise ValueError("n_candles must be either int or dict")


# --- Modified preprocess_csv ---
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

    # --- Apply feature pipeline ---
    extra_dicts = {}   # collect separatable feature groups
    if feature_pipeline is not None:
        for func in feature_pipeline:
            out = func(df)
            if isinstance(out, tuple):
                df, extra = out
                for k, v in extra.items():
                    extra_dicts[k] = v
            else:
                df = out

    # --- Handle dataset creation ---
    if isinstance(n_candles, int):
        # Normal supervised creation
        feature_cols = [col for col in df.columns if col not in ("timestamp", "label")]
        X, y = create_supervised(df, n_candles, feature_cols)

    elif isinstance(n_candles, dict):
        X_dict = {}
        Ys = []   # collect labels from each group

        for key, seq_len in n_candles.items():
            if key == "main":
                feature_cols = [col for col in df.columns if col not in ("timestamp", "label")]
                X_part, y_main = create_supervised(df, seq_len, feature_cols)
                X_dict[key] = X_part
                Ys.append(y_main)
            else:
                if key not in extra_dicts:
                    raise ValueError(f"Feature group '{key}' not found in pipeline outputs.")
                sub_df = extra_dicts[key]
                feature_cols = [c for c in sub_df.columns]
                X_part, y_sub = create_supervised(sub_df, seq_len, feature_cols, labels=df['label'].values)
                X_dict[key] = X_part
                Ys.append(y_sub)

        # ⚡ align everything by shortest length
        min_len = min(len(arr) for arr in Ys)
        for k in X_dict:
            X_dict[k] = X_dict[k][-min_len:]
        y = Ys[0][-min_len:]   # all Ys should be aligned now
        X = X_dict

    else:
        raise ValueError("n_candles must be int or dict")

    # --- Encode labels ---
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # --- Return dataset ---
    if isinstance(n_candles, dict):
        dataset = MultiInputDataset(X, y_encoded)
        if val_split:
            idx_train, idx_val = train_test_split(
                np.arange(len(y_encoded)),
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded
            )
            X_train_dict = {k: v[idx_train] for k, v in X.items()}
            X_val_dict = {k: v[idx_val] for k, v in X.items()}
            y_train, y_val = y_encoded[idx_train], y_encoded[idx_val]
            return (
                MultiInputDataset(X_train_dict, y_train),
                MultiInputDataset(X_val_dict, y_val),
                label_encoder,
                df,
            )
        else:
            return dataset, label_encoder, df

    else:
        # single input case (same as your old version)
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
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y_encoded, dtype=torch.long)
            dataset = TensorDataset(X_tensor, y_tensor)
            if val_split:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_tensor, y_tensor,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_encoded
                )
                return (
                    TensorDataset(X_train, y_train),
                    TensorDataset(X_val, y_val),
                    label_encoder,
                    df,
                    feature_cols
                )
            else:
                return dataset, label_encoder, df, feature_cols
