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
    val_split=True,
    test_size=0.2,
    random_state=42,
    for_xgboost=False,
    debug_sample=False,
    preserve_order=True,
    scale_labels = False,
    window_norm_fit = True,
    num_kernels=100,
    normalise=False,
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

    # --- Optionally fit target scalers ---
    if scale_labels and feature_pipeline is not None:
        feature_pipeline.fit_y(df_labels, lineprice_cols)

    # --- Collect sequences ---
    X_dicts_list = []
    y_list = []
    x_lengths = []
    label_lengths = []
    feature_cols = None
    feature_columns = {} 
    for _, row in df_labels.iterrows():
        # --- Slice main per label ---
        mask = (
            (feature_pipeline.main_data['timestamp'] >= row['startTime']) &
            (feature_pipeline.main_data['timestamp'] <= row['endTime'])
        )
        df_main = feature_pipeline.main_data.loc[mask].copy()

        # --- Start with main and slice global dicts using the same mask ---
        subseqs = {"main": df_main.copy()}
        if feature_pipeline is not None:
            for k, v in feature_pipeline.global_dicts.items():
                # slice extra dicts by the same time mask as main
                subseqs[k] = v.loc[mask].copy()

            # Apply per-window steps (optional)
            subseqs = feature_pipeline.apply_window(subseqs)
            if subseqs is None:
                continue  # skip bad indices

            # Normalize (fit=False),
            subseqs = feature_pipeline._normalize(subseqs, fit=False)
        # --- Collect features for all dicts ---
        X_dict = {}
        for dict_name, df_sub in subseqs.items():
            feats = [c for c in df_sub.columns if c != "timestamp"]
            # store feature names for later (first time or overwrite is fine if consistent)
            if dict_name not in feature_columns:
                feature_columns[dict_name] = feats.copy()

            if dict_name == "main" and feature_cols is None:
                feature_cols = feats

            arr = df_sub[feats].values.astype(np.float32)
            X_dict[dict_name] = arr

            # --- DEBUG print ---
            # print(f"[DEBUG] {dict_name}: df_sub shape = {df_sub.shape}, arr shape = {arr.shape}")

        if not X_dict or arr.shape[0] == 0:
            continue

        X_dicts_list.append(X_dict)
        x_lengths.append(len(subseqs["main"]))

        # --- Labels ---
        line_prices = row[lineprice_cols].fillna(0).values.astype(np.float32)
        if scale_labels and feature_pipeline is not None:
            row_df = pd.DataFrame([row[lineprice_cols].values], columns=lineprice_cols)
            row_scaled = feature_pipeline.transform_y(row_df, lineprice_cols).iloc[0].values.astype(np.float32)
            line_prices = row_scaled

        if preserve_order:
            y_list.append(line_prices)
            label_lengths.append((line_prices != 0).sum())
        else:
            nonzero_vals = line_prices[line_prices != 0]
            line_prices = np.sort(nonzero_vals)
            y_list.append(line_prices)
            label_lengths.append(line_prices.shape[0])


    # --- APPLY WINDOW NORMALIZATION AFTER WE COLLECT ALL SEQUENCES ---
    if feature_pipeline is not None and getattr(feature_pipeline, "window_norms", None):
        # if window_norm_fit==True -> fit scalers now (training)
        # if window_norm_fit==False -> only transform using previously fitted scalers (inference)
        X_dicts_list = feature_pipeline.apply_window_normalization(
            X_dicts_list, feature_columns, fit=window_norm_fit
        )

    # --- Pad labels ---
    max_len_y = max((len(arr) for arr in y_list), default=0)
    y = np.zeros((len(y_list), max_len_y), dtype=np.float32)
    for i, arr in enumerate(y_list):
        y[i, :len(arr)] = arr

    # ===============================
    # XGBoost mode
    # ===============================
    if for_xgboost:
        from utils.padd_sequence_xgboost import pad_sequences_dicts
        from sktime.transformations.panel.rocket import Rocket
        # Pad sequences
        X_main = pad_sequences_dicts(X_dicts_list, dict_name="main", strategy="forward_fill")
        
        X_main_rocket = np.transpose(X_main, (0, 2, 1))  
        print("X_main shape:", X_main_rocket.shape)  # should be (n_samples, time_length, n_channels)
        # --- Apply ROCKET ---
        rocket = Rocket(num_kernels=num_kernels,normalise=normalise, n_jobs=-1, random_state=random_state)
        X_rocket = rocket.fit_transform(X_main_rocket).values   # shape: (n_samples, n_rocket_features)

        # --- Debug print ---
        if debug_sample is not False:
            print("\n=== DEBUG SAMPLE CHECK (XGBoost mode) ===")
            indices = [0] if debug_sample is True else (
                [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
            )
            for idx in indices:
                print(f"\n--- Sequence {idx} ---")
                print("Label:", y_list[idx], "Encoded (padded):", y[idx])
                print("Shape:", X_dicts_list[idx]["main"].shape)
                print("First few rows of sequence:\n", X_dicts_list[idx]["main"][:])
                print("ROCKET feature vector (X_rocket):", X_rocket[idx])
        # --- Return splits or full dataset ---
        if val_split:
            idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_state)
            X_train_split, X_val_split = X_rocket[idx_train], X_rocket[idx_val]
            y_train_split, y_val_split = y[idx_train], y[idx_val]
            label_lengths_arr = np.array(label_lengths)
            train_length, test_length = label_lengths_arr[idx_train], label_lengths_arr[idx_val]
            return (
                X_train_split, y_train_split,
                X_val_split,   y_val_split,
                df_labels, feature_columns, max_len_y, train_length, test_length
            )
        else:
            return X_rocket, y, df_labels, feature_columns, max_len_y, label_lengths

    # ===============================
    # Torch Dataset mode
    # ===============================
    dataset = MultiInputDataset(
        {k: [d[k] for d in X_dicts_list] for k in X_dicts_list[0]},
        y,
        x_lengths
    )

    # --- Debug print ---
    if debug_sample is not False:
        print("\n=== DEBUG SAMPLE CHECK (Torch mode) ===")
        indices = [0] if debug_sample is True else (
            [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
        )
        for idx in indices:
            print(f"\n--- Sequence {idx} ---")
            print("Label:", y_list[idx], "Encoded (padded):", y[idx])
            for dict_name, arr in X_dicts_list[idx].items():
                print(f"[{dict_name}] Shape:", arr.shape)
                print(f"[{dict_name}] First few rows:\n", arr[:])
        print("==========================\n")

    if val_split:
        idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_state)
        X_train = {k: [X_dicts_list[i][k] for i in idx_train] for k in X_dicts_list[0]}
        X_val   = {k: [X_dicts_list[i][k] for i in idx_val]   for k in X_dicts_list[0]}
        return (
            MultiInputDataset(X_train, y[idx_train], [x_lengths[i] for i in idx_train]),
            MultiInputDataset(X_val,   y[idx_val],   [x_lengths[i] for i in idx_val]),
            df_labels, feature_columns, max_len_y
        )
    else:
        return dataset, df_labels, feature_columns, max_len_y
