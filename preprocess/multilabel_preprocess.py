import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# --- MultiInputDataset for dict of sequences ---
class MultiInputDataset(Dataset):
    def __init__(self, X_dict, y):
        """
        X_dict: dict of {feature_group_name: np.array of shape (num_samples, seq_len, feat_dim)}
        y: np.array of multi-hot encoded labels, shape (num_samples, num_classes)
        """
        self.X_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in X_dict.items()}
        self.y = torch.tensor(y, dtype=torch.float32)  # multi-label → float
        self.length = len(y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.X_dict.items()}
        return sample, self.y[idx]


# --- Modified create_supervised (multi-label safe) ---
def create_supervised(df, n_candles, feature_cols=None, labels=None):
    """
    If n_candles is an int: normal behavior
    If n_candles is dict: must be handled outside
    labels: array-like, can contain multi-labels (lists or arrays)
    """
    if isinstance(n_candles, int):
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in ("timestamp", "label")]

        X, y = [], []
        for i in range(n_candles, len(df)):
            label = df.loc[i, "label"] if "label" in df.columns else labels[i]

            # --- FIX for multilabel ---
            if label is None:
                continue
            if isinstance(label, (list, np.ndarray)):
                if len(label) == 0:   # skip empty multilabel
                    continue
            elif pd.isna(label):
                continue

            past_candles = df.loc[i-n_candles+1:i, feature_cols].values
            X.append(past_candles)
            y.append(label)

        return np.array(X), np.array(y, dtype=object)

    elif isinstance(n_candles, dict):
        raise ValueError("Dict mode must be handled outside, per feature group.")

    else:
        raise ValueError("n_candles must be either int or dict")


# --- Multi-label preprocess_csv ---
def preprocess_csv_multilabel(
    data_csv,
    labels_csv,
    n_candles=3,
    val_split=False,
    test_size=0.2,
    random_state=42,
    for_xgboost=False,
    feature_pipeline=None,
    debug_sample=False
):
    """
    Preprocess OHLCV data + labels into supervised format for MULTI-LABEL classification.
    """

    # --- Load OHLC data ---
    df_data = pd.read_csv(data_csv)
    if not all(col in df_data.columns for col in ["open", "high", "low", "close"]):
        df_data["open"] = df_data["close"]
        df_data["high"] = df_data["close"]
        df_data["low"] = df_data["close"]

    # --- Load labels ---
    df_labels = pd.read_csv(labels_csv)

    if "labels" in df_labels.columns:  # rename if needed
        df_labels = df_labels.rename(columns={"labels": "label"})
    # in preprocess_csv_multilabel.py

    # --- Normalize column names ---
    df_data.rename(columns={"time": "timestamp"}, inplace=True, errors="ignore")
    df_labels.rename(columns={"time": "timestamp"}, inplace=True, errors="ignore")

    # --- Ensure both timestamps are datetime ---
    # If df_data already has datetime-like strings ("2018-01-01"), convert to datetime
    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"], errors="coerce")

    # If df_labels has unix seconds, convert to datetime
    df_labels["timestamp"] = pd.to_datetime(df_labels["timestamp"], unit="s", errors="coerce")

    # Merge on timestamp
    df = pd.merge(df_data, df_labels[["timestamp", "label"]], on="timestamp", how="left")

    # --- Ensure labels are lists (multi-label) ---
    df["label"] = df["label"].apply(
        lambda x: eval(x) if isinstance(x, str) and (x.startswith("[") or "," in x) else [x]
    )

    # --- Apply feature pipeline ---
    extra_dicts = {}
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
        feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
        X, y_raw = create_supervised(df, n_candles, feature_cols)
    elif isinstance(n_candles, dict):
        X_dict = {}
        Ys = []
        for key, seq_len in n_candles.items():
            if key == "main":
                feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
                X_part, y_main = create_supervised(df, seq_len, feature_cols)
                X_dict[key] = X_part
                Ys.append(y_main)
            else:
                if key not in extra_dicts:
                    raise ValueError(f"Feature group '{key}' not found in pipeline outputs.")
                sub_df = extra_dicts[key]
                feature_cols = list(sub_df.columns)
                X_part, y_sub = create_supervised(sub_df, seq_len, feature_cols, labels=df["label"].values)
                X_dict[key] = X_part
                Ys.append(y_sub)

        min_len = min(len(arr) for arr in Ys)
        for k in X_dict:
            X_dict[k] = X_dict[k][-min_len:]
        y_raw = Ys[0][-min_len:]
        X = X_dict
    else:
        raise ValueError("n_candles must be int or dict")

    # --- Multi-label binarization ---
    def normalize_labels(y_raw):
        cleaned = []
        for lbl in y_raw:
            if lbl is None or (isinstance(lbl, float) and np.isnan(lbl)):
                cleaned.append([])  # no label
            elif isinstance(lbl, (list, np.ndarray)):
                cleaned.append([str(x) for x in lbl if pd.notna(x)])  # ensure list of strings
            else:
                cleaned.append([str(lbl)])  # wrap scalar as list
        return cleaned

    y_clean = normalize_labels(y_raw)

    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y_clean)

    # --- DEBUG SAMPLE PRINT ---
    if debug_sample is not False:
        print("\n=== DEBUG SAMPLE CHECK (MULTI-LABEL) ===")
        indices = [0] if debug_sample is True else list(debug_sample)
        for idx in indices:
            if isinstance(n_candles, int):
                seq_df = df.iloc[idx:idx+n_candles].copy()
                print("\n--- Sample index", idx, "---")
                print("Features (sequence):")
                print(seq_df[["timestamp"] + feature_cols])
                print("Labels:", y_raw[idx])
                print("Encoded:", y_encoded[idx])
            elif isinstance(n_candles, dict):
                print("\n--- Sample index", idx, "---")
                for k, v in X.items():
                    print(f"Feature group {k}: shape {v[idx].shape}")
                print("Labels:", y_raw[idx])
                print("Encoded:", y_encoded[idx])
        print("==========================\n")

    # --- RETURN ---
    if for_xgboost and isinstance(n_candles, int):
        X_flat = np.array([seq.flatten() for seq in X])
        if val_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y_encoded, test_size=test_size, random_state=random_state
            )
            return X_train, X_val, y_train, y_val, mlb, df, feature_cols  # 7 values
        else:
            return X_flat, None, y_encoded, None, mlb, df, feature_cols  # 7 values

    if isinstance(n_candles, dict):
        dataset = MultiInputDataset(X, y_encoded)
        if val_split:
            idx_train, idx_val = train_test_split(
                np.arange(len(y_encoded)), test_size=test_size, random_state=random_state
            )
            X_train_dict = {k: v[idx_train] for k, v in X.items()}
            X_val_dict = {k: v[idx_val] for k, v in X.items()}
            return (
                MultiInputDataset(X_train_dict, y_encoded[idx_train]),
                MultiInputDataset(X_val_dict, y_encoded[idx_val]),
                mlb,
                df,
            )
        else:
            return dataset, mlb, df
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        if val_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor, test_size=test_size, random_state=random_state
            )
            return (
                TensorDataset(X_train, y_train),
                TensorDataset(X_val, y_val),
                mlb,
                df,
                feature_cols,
            )
        else:
            return dataset, mlb, df, feature_cols
