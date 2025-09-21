import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from utils.weightening import get_label_weights

class MultiInputDataset(Dataset):
    def __init__(self, X_dict, y, x_lengths=None):
        self.X_dict = X_dict
        self.y = torch.tensor(y, dtype=torch.float32)
        self.x_lengths = torch.tensor(x_lengths, dtype=torch.long) if x_lengths is not None else None
        self.length = len(y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.X_dict.items()}
        if self.x_lengths is not None:
            return sample, self.y[idx], self.x_lengths[idx]
        else:
            return sample, self.y[idx]


def preprocess_csv_multilabel(
    data_csv,
    labels_csv,
    n_candles=3,
    feature_pipeline=None,
    val_split=True,
    test_size=0.2,
    random_state=42,
    for_xgboost=False,
    debug_sample=False,
    label_weighting="none"
):
    # --- Load Data ---
    df_data = pd.read_csv(data_csv)
    df_labels = pd.read_csv(labels_csv)

    # Normalize column names
    df_data.rename(columns={"time": "timestamp"}, inplace=True, errors="ignore")
    df_labels.rename(columns={"time": "timestamp"}, inplace=True, errors="ignore")

    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"], errors="coerce")
    df_labels["timestamp"] = pd.to_datetime(df_labels["timestamp"], unit="s", errors="coerce")
    # Merge
    df = pd.merge(df_data, df_labels[["timestamp", "label"]], on="timestamp", how="left")

    # Ensure multi-label format
    df["label"] = df["label"].apply(
        lambda x: eval(x) if isinstance(x, str) and (x.startswith("[") or "," in x) else [x]
    )
    # --- Apply feature pipeline ---
    if feature_pipeline is not None:
        feature_pipeline.fit(df)   # optional: only if training
        # subseqs = feature_pipeline.apply_window({"main": df})
        # subseqs = feature_pipeline._normalize(subseqs, fit=False)
        df = feature_pipeline.global_data["main"]
    # --- Collect sequences ---
    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    X, y_raw = [], []
    for i in range(n_candles, len(df)):
        past_candles = df.iloc[i-n_candles+1:i+1][feature_cols].values
        lbl = df.iloc[i]["label"]
        if lbl is not None and len(lbl) > 0 and any(pd.notna(x) for x in lbl):
            X.append(past_candles.astype(np.float32))
            y_raw.append(lbl)

    print("Collected sequences:", len(X))
    print("y_raw sample:", y_raw[:10])

    # --- Encode labels ---
    def check_labels(y_raw):
        return [[str(x) for x in (lbl if isinstance(lbl, (list, np.ndarray)) else [lbl]) if pd.notna(x)]
                for lbl in y_raw]

    y_clean = check_labels(y_raw)
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y_clean)

    # --- Compute label weights ---
    label_weights = get_label_weights(y_encoded, mlb, label_weighting)
    # --- Torch/XGB mode ---
    X_arr = np.array(X, dtype=np.float32)
    # --- Debug print ---
    if debug_sample is not False:
        print("\n=== DEBUG SAMPLE CHECK ===")
        indices = [0] if debug_sample is True else (
            [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
        )
        for idx in indices:
            print(f"\n--- Sequence {idx} ---")
            print("Original label(s):", y_raw[idx])
            print("Cleaned label(s):", y_clean[idx])
            print("Encoded:", y_encoded[idx])
            print("Feature shape:", X_arr[idx].shape)
            print("First few timesteps:\n", X_arr[idx][:])  # show   candles
        print("==========================\n")
    if for_xgboost:
        X_flat = X_arr.reshape(X_arr.shape[0], -1)
        if val_split:
            X_train, X_val, y_train, y_val = train_test_split(X_flat, y_encoded, test_size=test_size, random_state=random_state)
            return X_train, y_train, X_val, y_val, df_labels, feature_cols, mlb, label_weights
        else:
            return X_flat, y_encoded, df_labels, feature_cols, mlb, label_weights
    else:
        dataset = TensorDataset(torch.tensor(X_arr), torch.tensor(y_encoded, dtype=torch.float32))
        if val_split:
            idx_train, idx_val = train_test_split(np.arange(len(y_encoded)), test_size=test_size, random_state=random_state)
            return (
                TensorDataset(torch.tensor(X_arr[idx_train]), torch.tensor(y_encoded[idx_train])),
                TensorDataset(torch.tensor(X_arr[idx_val]), torch.tensor(y_encoded[idx_val])),
                df_labels, feature_cols, mlb, label_weights
            )
        else:
            return dataset, df_labels, feature_cols, mlb, label_weights
