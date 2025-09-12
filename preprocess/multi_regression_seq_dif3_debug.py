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
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    # --- Load data ---
    df_data = pd.read_csv(data_csv)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])

    df_labels = pd.read_csv(labels_csv)
    df_labels['startTime'] = pd.to_datetime(df_labels['startTime'], unit='s')
    df_labels['endTime']   = pd.to_datetime(df_labels['endTime'], unit='s')
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp']).dt.normalize()
    df_labels['startTime'] = df_labels['startTime'].dt.normalize()
    df_labels['endTime']   = df_labels['endTime'].dt.normalize()

    lineprice_cols = [c for c in df_labels.columns if c.startswith("linePrice")]
    print(df_data['timestamp'].head())
    print(df_labels['startTime'].head())
    print(df_labels['endTime'].head())
    # --- Fit pipeline globally ---
    if feature_pipeline is not None:
        feature_pipeline.fit(df_data)

    # --- Collect sequences ---
    X_dicts_list = []   # list of dicts (main + extra dicts)
    y_list = []
    x_lengths = []
    label_lengths = []
    feature_cols = None

    for idx_label, row in df_labels.iterrows():
        # tmask = (df_data['timestamp'] >= row['startTime'])
        # print ("its ok start",tmask)
        # tmask = (df_data['timestamp'] <= row['endTime'])
        # print ("its ok end",tmask)
        mask = (df_data['timestamp'] >= row['startTime']) & (df_data['timestamp'] <= row['endTime'])
        # print(df_data['timestamp'],"data ts")
        # print(row['startTime'],"data start")
        # print(row['endTime'] ,"data end")
        df_main = df_data.loc[mask].copy()
        print(f"[DEBUG] Label {row['startIndex']}:{row['endIndex']} - timestamp mask rows: {mask.sum()}")

        if df_main.empty:
            print(f"[SKIP] Empty df_main before pipeline for label {row['startIndex']}:{row['endIndex']}")
            continue

        subseqs = {}
        subseqs["main"] = df_main

        # --- Apply per-window steps ---
        if feature_pipeline is not None:
            df_main_windowed = feature_pipeline.apply_window(df_main)
            print(f"[DEBUG] After per-window steps - df_main rows: {len(df_main_windowed)}")
            subseqs["main"] = df_main_windowed

        # --- Apply separatable dict steps ---
        # Here we check for any other dicts added by make_step with separatable logic
        if feature_pipeline is not None:
            for step, per_win in zip(feature_pipeline.steps, feature_pipeline.per_window_flags):
                if hasattr(step, "_separatable") and step._separatable:
                    out = step(subseqs["main"], **getattr(step, "_step_kwargs", {}))
                    if isinstance(out, tuple):
                        df_new, _ = out
                    else:
                        df_new = out
                    dict_name = getattr(step, "_dict_name", "extra")
                    subseqs[dict_name] = df_new
                    print(f"[DEBUG] Added dict '{dict_name}' - rows: {len(df_new)}")

        # --- Normalize each dict ---
        for dict_name, subseq in subseqs.items():
            norm_cfg = feature_pipeline.norm_methods.get(dict_name, {}) if feature_pipeline else {}
            if norm_cfg:
                subseq_norm = feature_pipeline._normalize_single(
                    subseq, norm_cfg, fit=False, dict_name=dict_name
                )
                subseqs[dict_name] = subseq_norm
                print(f"[DEBUG] After normalization '{dict_name}' - rows: {len(subseq_norm)}, NaNs: {subseq_norm.isna().sum().sum()}")

        # --- Collect features ---
        X_dict = {}
        skip_seq = False
        for dict_name, subseq in subseqs.items():
            feats = [c for c in subseq.columns if c != "timestamp"]
            arr = subseq[feats].values.astype(np.float32)
            if arr.shape[0] == 0:
                print(f"[SKIP] Sequence empty in dict '{dict_name}' for label {row['startIndex']}:{row['endIndex']}")
                skip_seq = True
            X_dict[dict_name] = arr
            if dict_name == "main" and feature_cols is None:
                feature_cols = feats

        if skip_seq:
            continue

        X_dicts_list.append(X_dict)
        x_lengths.append(len(subseqs["main"]))

        # --- Labels ---
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

    print(f"[DEBUG] Total sequences collected: {len(X_dicts_list)}")
    return X_dicts_list, y, x_lengths, label_lengths, feature_cols
