import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import ast


class MultiInputDataset(Dataset):
    def __init__(self, X_dict, y):
        self.X_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in X_dict.items()}
        self.y = torch.tensor(y, dtype=torch.float32)   # regression targets
        self.length = len(y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.X_dict.items()}
        return sample, self.y[idx]


def preprocess_sequences_csv_multilines(
    data_csv,
    labels_csv,
    feature_pipeline=None,
    val_split=False,
    test_size=0.2,
    random_state=42,
    for_xgboost=False,
    debug_sample=False,
    n_candles=None
):
    """
    Preprocess main data + linePrices sequences from labels_csv for variable-length multi-line regression.
    """

    # --- Load main CSV ---
    df_data = pd.read_csv(data_csv)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])

    # --- Load labels ---
    df_labels = pd.read_csv(labels_csv)
    df_labels['startTime'] = pd.to_datetime(df_labels['startTime'], unit='s')
    df_labels['endTime'] = pd.to_datetime(df_labels['endTime'], unit='s')

    # Parse linePrices if stored as string
    if isinstance(df_labels.loc[0, 'linePrices'], str):
        df_labels['linePrices'] = df_labels['linePrices'].apply(ast.literal_eval)

    # --- Apply feature pipeline if any ---
    extra_dicts = {}
    df_main = df_data.copy()
    if feature_pipeline is not None:
        for func in feature_pipeline:
            out = func(df_main)
            if isinstance(out, tuple):
                df_main, extra = out
                extra_dicts.update(extra)
            else:
                df_main = out

    # --- Build sequences ---
    X_list, y_list = [], []
    X_dict_list = {k: [] for k in (extra_dicts.keys() if n_candles else [])}

    for _, row in df_labels.iterrows():
        subseq = df_main.iloc[row['startIndex']:row['endIndex'] + 1]
        main_feats = [c for c in subseq.columns if c != 'timestamp']
        main_seq = subseq[main_feats].values

        line_prices = np.array(row['linePrices'], dtype=np.float32)

        X_list.append(main_seq)
        y_list.append(line_prices)

        if n_candles:
            for key, sub_df in extra_dicts.items():
                X_dict_list[key].append(sub_df.iloc[row['startIndex']:row['endIndex'] + 1].values)

    # --- Pad y_list (linePrices targets) ---
    max_len_y = max(len(arr) for arr in y_list) 
    y = np.zeros((len(y_list), max_len_y), dtype=np.float32)
    for i, arr in enumerate(y_list):
        y[i, :len(arr)] = arr

    # --- Pad sequences helper ---
    def pad_sequences(seq_list):
        max_len = max(len(s) for s in seq_list)
        feat_dim = seq_list[0].shape[1]
        out = np.zeros((len(seq_list), max_len, feat_dim), dtype=np.float32)
        for i, s in enumerate(seq_list):
            out[i, :len(s), :] = s
        return out

    X_main = pad_sequences(X_list)

    if n_candles:
        X_dict = {"main": X_main}
        for k, seqs in X_dict_list.items():
            X_dict[k] = pad_sequences(seqs)
    else:
        X_dict = None

    feature_cols = ["linePrices"]

    # ======================================================
    # --- Debug sample print ---
    if debug_sample is not False:
        print("\n=== DEBUG SAMPLE CHECK ===")
        indices = [0] if debug_sample is True else (
            [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
        )
        for idx in indices:
            print(f"\n--- Sequence {idx} ---")
            print("Label (linePrices padded):", y[idx])
            if n_candles:
                for k, v in X_dict.items():
                    print(f"Feature group: {k}, Shape: {v[idx].shape}")
                    print(v[idx][:5])
            else:
                print("Shape:", X_main[idx].shape)
                print("First few rows:\n", X_main[idx][:5])
        print("==========================\n")
    # ======================================================

    # ======================================================
    # --- For XGBoost mode ---
    if for_xgboost:
        if n_candles:
            X_flat_dict = {k: np.array([seq.flatten() for seq in v]) for k, v in X_dict.items()}
            if val_split:
                idx_train, idx_val = train_test_split(
                    np.arange(len(y)), test_size=test_size, random_state=random_state
                )
                X_train_dict = {k: v[idx_train] for k, v in X_flat_dict.items()}
                X_val_dict = {k: v[idx_val] for k, v in X_flat_dict.items()}
                return (X_train_dict, y[idx_train],
                        X_val_dict, y[idx_val],
                        df_labels, feature_cols)
            else:
                return X_flat_dict, y, df_labels, feature_cols
        else:
            X_flat = np.array([seq.flatten() for seq in X_main])
            if val_split:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_flat, y,
                    test_size=test_size,
                    random_state=random_state
                )
                return X_train, y_train, X_val, y_val, df_labels, feature_cols,max_len_y
            else:
                return X_flat, y, df_labels, feature_cols,max_len_y
    # ======================================================

    # --- Create Torch dataset ---
    if n_candles:
        dataset = MultiInputDataset(X_dict, y)
    else:
        X_tensor = torch.tensor(X_main, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)

    # --- Train/Val split ---
    if val_split:
        idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=test_size,
                                             random_state=random_state)
        if n_candles:
            X_train_dict = {k: v[idx_train] for k,v in X_dict.items()}
            X_val_dict   = {k: v[idx_val]   for k,v in X_dict.items()}
            return (MultiInputDataset(X_train_dict, y[idx_train]),
                    MultiInputDataset(X_val_dict, y[idx_val]),
                    df_labels, feature_cols, max_len_y)
        else:
            X_train = X_tensor[idx_train]
            X_val   = X_tensor[idx_val]
            y_train = y_tensor[idx_train]
            y_val   = y_tensor[idx_val]
            return (TensorDataset(X_train, y_train),
                    TensorDataset(X_val, y_val),
                    df_labels, feature_cols, max_len_y)
    else:
        return dataset, df_labels, feature_cols, max_len_y
