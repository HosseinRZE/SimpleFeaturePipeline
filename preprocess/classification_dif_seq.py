import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# --- MultiInputDataset stays the same ---
class MultiInputDataset(Dataset):
    def __init__(self, X_dict, y):
        self.X_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in X_dict.items()}
        self.y = torch.tensor(y, dtype=torch.long)
        self.length = len(y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.X_dict.items()}
        return sample, self.y[idx]


# --- New: preprocess based on startTime–endTime sequences ---
def preprocess_sequences_csv(
    data_csv,
    labels_csv,
    feature_pipeline=None,
    val_split=False,
    test_size=0.2,
    random_state=42,
    for_xgboost=False,
    debug_sample=False,
    n_candles=None,   # dict or None (kept for multi-input consistency)
):
    """
    Preprocess OHLCV data into sequences defined by labels_csv (startTime–endTime).

    Args:
        data_csv (str): CSV with OHLCV data. Must include 'timestamp' and 'close'.
        labels_csv (str): CSV with labeled sequences. Must include ['startTime','endTime','label'].
        feature_pipeline (list of callables, optional): Feature transforms like before.
        val_split (bool): Whether to return train/val split.
        test_size (float): Fraction for validation.
        for_xgboost (bool): If True, sequences are flattened (careful with variable lengths).
        debug_sample (bool/int/list): Show sample(s) for debugging.
        n_candles (dict or None): If dict, expects multi-input feature groups (keys = group names).

    Returns:
        Same structure as old preprocess_csv:
            - Single input: TensorDataset/arrays
            - Multi-input: MultiInputDataset/tuple of them
            - Always returns label_encoder + processed df
    """

    # --- Load OHLCV data ---
    df_data = pd.read_csv(data_csv)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
    if not all(col in df_data.columns for col in ['open', 'high', 'low', 'close']):
        df_data['open'] = df_data['close']
        df_data['high'] = df_data['close']
        df_data['low'] = df_data['close']

    # --- Load labels with start/end time ---
    df_labels = pd.read_csv(labels_csv)
    df_labels['startTime'] = pd.to_datetime(df_labels['startTime'], unit='s')
    df_labels['endTime'] = pd.to_datetime(df_labels['endTime'], unit='s')

    # --- Apply feature pipeline ---
    extra_dicts = {}
    df = df_data.copy()
    if feature_pipeline is not None:
        for func in feature_pipeline:
            out = func(df)
            if isinstance(out, tuple):
                df, extra = out
                extra_dicts.update(extra)
            else:
                df = out

    # --- Build sequences ---
    X_list, y_list = [], []
    X_dict_list = {k: [] for k in (extra_dicts.keys() if n_candles else [])}

    for _, row in df_labels.iterrows():
        start, end, label = row['startTime'], row['endTime'], row['label']

        # main feature group
        subseq = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        if subseq.empty:
            continue
        main_feats = [c for c in subseq.columns if c not in ("timestamp", "label")]
        X_list.append(subseq[main_feats].values)
        y_list.append(label)

        # other feature groups
        if n_candles:
            for key, sub_df in extra_dicts.items():
                sub_seq = sub_df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
                if sub_seq.empty:
                    raise ValueError(f"Empty subsequence for feature group '{key}' in range {start}–{end}")
                X_dict_list[key].append(sub_seq.values)

    # --- Encode labels ---
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_list)

    # --- Padding (for Torch tensors) ---
    def pad_sequences(seq_list):
        max_len = max(len(s) for s in seq_list)
        feat_dim = seq_list[0].shape[1]
        out = np.zeros((len(seq_list), max_len, feat_dim), dtype=np.float32)
        for i, s in enumerate(seq_list):
            out[i, :len(s), :] = s
        return out

    # main
    X_main = pad_sequences(X_list)

    if n_candles:  # multi-input
        X_dict = {"main": X_main}
        for k, seqs in X_dict_list.items():
            X_dict[k] = pad_sequences(seqs)
    else:
        X_dict = None

    # --- Debug print ---
    if debug_sample is not False:
        print("\n=== DEBUG SAMPLE CHECK ===")
        indices = [0] if debug_sample is True else (
            [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
        )
        for idx in indices:
            print(f"\n--- Sequence {idx} ---")
            print("Label:", y_list[idx], "Encoded:", y_encoded[idx])
            if n_candles:
                for k, v in X_dict.items():
                    print(f"Feature group: {k}, Shape: {v[idx].shape}")
                    print(v[idx][:5])
            else:
                print("Shape:", X_list[idx].shape)
                print("First few rows:\n", X_list[idx][:5])
        print("==========================\n")

    # --- for_xgboost mode ---
    if for_xgboost:
        if n_candles:
            X_flat_dict = {k: np.array([seq.flatten() for seq in v]) for k, v in X_dict.items()}
            if val_split:
                idx_train, idx_val = train_test_split(
                    np.arange(len(y_encoded)),
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_encoded,
                )
                X_train_dict = {k: v[idx_train] for k, v in X_flat_dict.items()}
                X_val_dict = {k: v[idx_val] for k, v in X_flat_dict.items()}
                return (X_train_dict, y_encoded[idx_train],
                        X_val_dict, y_encoded[idx_val],
                        label_encoder, df)
            else:
                return X_flat_dict, y_encoded, label_encoder, df
        else:
            X_flat = np.array([seq.flatten() for seq in X_list])
            if val_split:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_flat, y_encoded,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_encoded
                )
                return X_train, y_train, X_val, y_val, label_encoder, df
            else:
                return X_flat, y_encoded, label_encoder, df

    # --- Torch mode ---
    if n_candles:
        dataset = MultiInputDataset(X_dict, y_encoded)
        feature_cols = list(df.columns.difference(['timestamp', 'label']))  # preserve feature names
        if val_split:
            idx_train, idx_val = train_test_split(
                np.arange(len(y_encoded)),
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded
            )
            X_train_dict = {k: v[idx_train] for k, v in X_dict.items()}
            X_val_dict = {k: v[idx_val] for k, v in X_dict.items()}
            return (
                MultiInputDataset(X_train_dict, y_encoded[idx_train]),
                MultiInputDataset(X_val_dict, y_encoded[idx_val]),
                label_encoder,
                df,
                feature_cols
            )
        else:
            return dataset, label_encoder, df, feature_cols
    else:
        X_tensor = torch.tensor(X_main, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        feature_cols = main_feats  # already defined when building X_main
        if val_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor,
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded,
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
