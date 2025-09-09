import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import ast
class MultiInputDataset(Dataset):
    def __init__(self, X_dict, y, x_lengths):
        """
        X_dict: dict mapping feature-group -> list of numpy arrays (variable-length per sample)
        y: padded numpy array (n_samples, max_len_y)
        x_lengths: list/array of true lengths for X (per-sample, before padding)
        """
        self.X_dict = X_dict  # keep lists of numpy arrays (one entry per sample)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        self.length = len(y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Convert the variable-length arrays to tensors on access
        sample = {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.X_dict.items()}
        return sample, self.y[idx], self.x_lengths[idx]

def preprocess_sequences_csv_multilines(
    data_csv,
    labels_csv,
    feature_pipeline=None,
    val_split=False,
    test_size=0.2,
    random_state=42,
    for_xgboost=False,
    debug_sample=False,
    preserve_order = False
):
    """
    Preprocess main data + linePrices sequences for multi-line regression.

    - Keeps variable-length X sequences (padding happens in collate_batch).
    - Computes both x_lengths (true lengths of input sequences) and
      label_lengths (true lengths of label sequences).
    - If for_xgboost=True, returns (X_train, y_train, X_val, y_val, df_labels, feature_cols, max_len_y, label_lengths)
      where label_lengths is the list of true label lengths for the whole dataset.
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

    # Fit pipeline globally (if provided)
    if feature_pipeline is not None:
        feature_pipeline.fit(df_data)

    df_main = df_data.copy()
    if feature_pipeline is not None:
        for step, per_window in zip(feature_pipeline.steps, feature_pipeline.per_window_flags):
            if not per_window:
                df_main = step(df_main)

    X_list = []
    y_list = []
    x_lengths = []      # true input sequence lengths for each sample
    label_lengths = []  # true label length (number of real label entries) for each sample
    real_feature_cols = None

    for _, row in df_labels.iterrows():
        subseq = df_main.iloc[row['startIndex']:row['endIndex'] + 1].copy()

        if feature_pipeline is not None:
            subseq = feature_pipeline.apply_window(subseq)

        main_feats = [c for c in subseq.columns if c != 'timestamp']
        if real_feature_cols is None:
            real_feature_cols = main_feats

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

        arr = subseq[main_feats].values.astype(np.float32)
        X_list.append(arr)
        x_lengths.append(arr.shape[0])
        if preserve_order:
            # labels (linePrices) — keep order, pad with -1.0 for missing
            line_prices = row[lineprice_cols].fillna(0).values.astype(np.float32)
            y_list.append(line_prices)
            label_lengths.append((line_prices != 0).sum())  # count valid labels
        else:
            # labels (linePrices) — these are already trimmed (dropna)
            line_prices = row[lineprice_cols].dropna().values.astype(np.float32)
            line_prices = np.sort(line_prices)
            y_list.append(line_prices)
            label_lengths.append(line_prices.shape[0])

    # Pad y globally (we keep labels padded for convenience)
    max_len_y = max(len(arr) for arr in y_list) if len(y_list) > 0 else 0
    y = np.zeros((len(y_list), max_len_y), dtype=np.float32)
    for i, arr in enumerate(y_list):
        y[i, :len(arr)] = arr

    # === XGBoost mode: return flattened features + label_lengths (true label lengths) ===
    if for_xgboost:
        # feature dimension:
        feat_dim = X_list[0].shape[1] if len(X_list) > 0 else 0
        # average-pool across time for each sample (if sample empty, zero vector)
        X_flat = np.stack([
            arr.mean(axis=0) if arr.shape[0] > 0 else np.zeros((feat_dim,), dtype=np.float32)
            for arr in X_list
        ])

        # --- Debug print ---
        if debug_sample is not False:
            print("\n=== DEBUG SAMPLE CHECK (XGBoost mode) ===")
            indices = [0] if debug_sample is True else (
                [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
            )
            for idx in indices:
                print(f"\n--- Sequence {idx} ---")
                print("Label:", y_list[idx], "Encoded (padded):", y[idx])
                print("Shape:", X_list[idx].shape)
                print("First few rows of sequence:\n", X_list[idx][:])
                print("Flattened feature vector (X_flat):", X_flat[idx])
            print("==========================\n")

        if val_split:
            idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=test_size,
                                                 random_state=random_state)
            return (X_flat[idx_train], y[idx_train],
                    X_flat[idx_val], y[idx_val],
                    df_labels, real_feature_cols, max_len_y, label_lengths)
        else:
            return (X_flat, y, df_labels, real_feature_cols, max_len_y, label_lengths)

    # === Torch dataset mode: keep variable-length X and return MultiInputDataset (x_lengths used) ===
    dataset = MultiInputDataset({"main": X_list}, y, x_lengths)

    # --- Debug print ---
    if debug_sample is not False:
        print("\n=== DEBUG SAMPLE CHECK (Torch mode) ===")
        indices = [0] if debug_sample is True else (
            [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
        )
        for idx in indices:
            print(f"\n--- Sequence {idx} ---")
            print("Label:", y_list[idx], "Encoded (padded):", y[idx])
            print("Shape:", X_list[idx].shape)
            print("First few rows of sequence:\n", X_list[idx][:5])
        print("==========================\n")

    if val_split:
        idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=test_size,
                                             random_state=random_state)
        X_train = {"main": [X_list[i] for i in idx_train]}
        X_val   = {"main": [X_list[i] for i in idx_val]}

        return (MultiInputDataset(X_train, y[idx_train], [x_lengths[i] for i in idx_train]),
                MultiInputDataset(X_val,   y[idx_val],   [x_lengths[i] for i in idx_val]),
                df_labels, real_feature_cols, max_len_y)
    else:
        return dataset, df_labels, real_feature_cols, max_len_y
