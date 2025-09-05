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
    Preprocess main data + linePrices sequences for multi-line regression.
    Tracks real columns after pipeline transformations.
    Normalizes BEFORE padding.
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    df_data = pd.read_csv(data_csv)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
    df_labels = pd.read_csv(labels_csv)
    df_labels['startTime'] = pd.to_datetime(df_labels['startTime'], unit='s')
    df_labels['endTime']   = pd.to_datetime(df_labels['endTime'], unit='s')
    lineprice_cols = [c for c in df_labels.columns if c.startswith("linePrice")]

    # --- MODIFICATION 1: FIT SCALER GLOBALLY ---
    # Fit the pipeline on the entire dataset first to learn global scaling params.
    if feature_pipeline is not None:
        feature_pipeline.fit(df_data)

    # --- Apply global pipeline steps (as before) ---
    df_main = df_data.copy()
    if feature_pipeline is not None:
        for step, per_window in zip(feature_pipeline.steps, feature_pipeline.per_window_flags):
            if not per_window:
                df_main = step(df_main)

    # --- Build sequences (loop) ---
    X_list, y_list = [], []
    real_feature_cols = None

    for _, row in df_labels.iterrows():
        subseq = df_main.iloc[row['startIndex']:row['endIndex'] + 1].copy()

        # Apply per-window steps (as before)
        if feature_pipeline is not None:
            subseq = feature_pipeline.apply_window(subseq)

        main_feats = [c for c in subseq.columns if c != 'timestamp']
        if real_feature_cols is None:
            real_feature_cols = main_feats

        # --- MODIFICATION 2: TRANSFORM SEQUENCES INDIVIDUALLY ---
        # Normalize using the pre-fitted global scalers.
        if feature_pipeline is not None:
            norm_cfg = feature_pipeline.norm_methods.get("main", {})
            norm_cols = [c for c in norm_cfg.keys() if c in subseq.columns]
            if norm_cols:
                subseq[norm_cols] = feature_pipeline._normalize_single(
                    subseq[norm_cols],
                    {k: v for k, v in norm_cfg.items() if k in norm_cols},
                    fit=False,  
                    dict_name="main"
                )

        X_list.append(subseq[main_feats].values)
        
        # --- Process linePrices (as before) ---
        line_prices = row[lineprice_cols].dropna().values.astype(np.float32)
        line_prices = np.sort(line_prices)
        y_list.append(line_prices)

    # --- True lengths and max_len_y ---
    seq_lengths_true = [len(arr) for arr in y_list]
    max_len_y = max(seq_lengths_true)

    # --- Pad y_list ---
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
    X_dict = {"main": X_main}

    # --- Debug sample ---
    if debug_sample:
        print("\n=== DEBUG SAMPLE ===")
        for idx in [0]:
            print("Label (linePrices padded):", y[idx])
            for k, v in X_dict.items():
                print(f"Feature group: {k}, Shape: {v[idx].shape}")
                print("Columns:", real_feature_cols)
                print(v[idx][:5])
        print("===================")

    # --- Torch dataset ---
    dataset = MultiInputDataset(X_dict, y)

    # --- Train/Val split ---
    if val_split:
        idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=test_size,
                                             random_state=random_state)
        X_train_dict = {k: v[idx_train] for k,v in X_dict.items()}
        X_val_dict   = {k: v[idx_val]   for k,v in X_dict.items()}
        return (MultiInputDataset(X_train_dict, y[idx_train]),
                MultiInputDataset(X_val_dict, y[idx_val]),
                df_labels, real_feature_cols, max_len_y)
    else:
        return dataset, df_labels, real_feature_cols, max_len_y
