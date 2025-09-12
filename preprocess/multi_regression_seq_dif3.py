import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset
import torch

# ===============================
# Preprocessing Function
# ===============================
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
    preserve_order=True
):
    """
    Preprocess main data + optional extra dicts for multi-line regression.

    - Supports FeaturePipeline with global dicts.
    - Applies global steps + normalization.
    - Per-window steps apply only to "main" (optional).
    - Drops bad indices consistently across all dicts.
    - Returns X_dicts / y ready for torch or X_flat for XGBoost.
    """

    # --- Load data ---
    df_data = pd.read_csv(data_csv)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])

    df_labels = pd.read_csv(labels_csv)
    df_labels['startTime'] = pd.to_datetime(df_labels['startTime'], unit='s')
    df_labels['endTime']   = pd.to_datetime(df_labels['endTime'], unit='s')
    lineprice_cols = [c for c in df_labels.columns if c.startswith("linePrice")]

    # --- Fit pipeline globally ---
    if feature_pipeline is not None:
        feature_pipeline.fit(df_data)

    # --- Collect sequences ---
    X_dicts_list = []
    y_list = []
    x_lengths = []
    label_lengths = []
    feature_cols = None

    for _, row in df_labels.iterrows():
        mask = (feature_pipeline.main_data['timestamp'] >= row['startTime']) & (feature_pipeline.main_data['timestamp'] <= row['endTime'])

        df_main = feature_pipeline.main_data.loc[mask].copy()

        # Start with global dicts
        subseqs = {"main": df_main.copy()}
        if feature_pipeline is not None:
            for k, v in feature_pipeline.global_dicts.items():
                subseqs[k] = v.copy()

            # Drop global invalid indices
            if feature_pipeline.global_invalid:
                for k in subseqs:
                    subseqs[k] = subseqs[k].drop(index=feature_pipeline.global_invalid, errors="ignore")
            # print("subseqs[k].head()",subseqs[k].head())
            # Apply per-window steps
            for step, per_win in zip(feature_pipeline.steps, feature_pipeline.per_window_flags):
                # print("subseqs[main]",subseqs["main"].head())
                if per_win:
                    result = step(subseqs["main"])
                    if len(result) == 2:
                        df_sub, bad_idx = result
                        subseqs["main"] = df_sub.drop(index=bad_idx)
                        for k in subseqs:
                            if k != "main":
                                subseqs[k] = subseqs[k].drop(index=bad_idx, errors="ignore")
                    elif len(result) == 3:
                        df_sub, bad_idx, dicts_new = result
                        subseqs["main"] = df_sub.drop(index=bad_idx)
                        for k, v in dicts_new.items():
                            subseqs[k] = v.drop(index=bad_idx, errors="ignore")
                    else:
                        raise ValueError("Step must return 2- or 3-tuple")

            # Normalize all dicts
            for dict_name, df_sub in subseqs.items():
                norm_cfg = feature_pipeline.norm_methods.get(dict_name, {})
                if norm_cfg:
                    subseqs[dict_name] = feature_pipeline._normalize_single(
                        df_sub, norm_cfg, fit=False, dict_name=dict_name
                    )

        # --- Collect features for all dicts ---
        X_dict = {}
        for dict_name, df_sub in subseqs.items():
            feats = [c for c in df_sub.columns if c != "timestamp"]
            if dict_name == "main" and feature_cols is None:
                feature_cols = feats
            arr = df_sub[feats].values.astype(np.float32)

            X_dict[dict_name] = arr
        if not X_dict or arr.shape[0] == 0:
            continue

        X_dicts_list.append(X_dict)
        x_lengths.append(len(subseqs["main"]))

        # Labels
        if preserve_order:
            line_prices = row[lineprice_cols].fillna(0).values.astype(np.float32)
            y_list.append(line_prices)
            label_lengths.append((line_prices != 0).sum())
        else:
            line_prices = row[lineprice_cols].dropna().values.astype(np.float32)
            line_prices = np.sort(line_prices)
            y_list.append(line_prices)
            label_lengths.append(line_prices.shape[0])

    # --- Pad labels ---
    max_len_y = max((len(arr) for arr in y_list), default=0)
    y = np.zeros((len(y_list), max_len_y), dtype=np.float32)
    for i, arr in enumerate(y_list):
        y[i, :len(arr)] = arr

    # ===============================
    # XGBoost mode
    # ===============================
    if for_xgboost:
        feat_dim = X_dicts_list[0]["main"].shape[1] if X_dicts_list else 0
        X_flat = np.stack([
            arr["main"].mean(axis=0) if arr["main"].shape[0] > 0 else np.zeros((feat_dim,), dtype=np.float32)
            for arr in X_dicts_list
        ])
        if val_split:
            idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_state)
            return (X_flat[idx_train], y[idx_train], X_flat[idx_val], y[idx_val], df_labels, feature_cols, max_len_y, label_lengths)
        else:
            return X_flat, y, df_labels, feature_cols, max_len_y, label_lengths

    # ===============================
    # Torch Dataset mode
    # ===============================
    dataset = MultiInputDataset(
        {k: [d[k] for d in X_dicts_list] for k in X_dicts_list[0]},
        y,
        x_lengths
    )

    if val_split:
        idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_state)
        X_train = {k: [X_dicts_list[i][k] for i in idx_train] for k in X_dicts_list[0]}
        X_val   = {k: [X_dicts_list[i][k] for i in idx_val] for k in X_dicts_list[0]}
        return MultiInputDataset(X_train, y[idx_train], [x_lengths[i] for i in idx_train]), \
               MultiInputDataset(X_val, y[idx_val], [x_lengths[i] for i in idx_val]), \
               df_labels, feature_cols, max_len_y
    else:
        return dataset, df_labels, feature_cols, max_len_y
