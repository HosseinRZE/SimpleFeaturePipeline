import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from sklearn.model_selection import train_test_split
from utils.build_multi_input_dataset import build_multiinput_dataset
    
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
    scale_labels=False,
    window_norm_fit=True,
):
    """
    Preprocess main data + optional extra dicts for multi-line regression.

    - Supports FeaturePipeline with global dicts.
    - Applies global steps + normalization.
    - Per-window steps apply only to "main" (optional).
    - Drops bad indices consistently across all dicts.
    - Returns dataset dicts + labels, ready for transformations.
    """

    # --- Load data ---
    df_data = pd.read_csv(data_csv)
    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"])

    df_labels = pd.read_csv(labels_csv)
    df_labels["startTime"] = pd.to_datetime(df_labels["startTime"], unit="s")
    df_labels["endTime"]   = pd.to_datetime(df_labels["endTime"], unit="s")
    lineprice_cols = [c for c in df_labels.columns if c.startswith("linePrice")]

    # --- Fit pipeline globally ---
    if feature_pipeline is not None:
        feature_pipeline.fit(df_data)

    # --- Optionally fit target scalers ---
    if scale_labels and feature_pipeline is not None:
        feature_pipeline.fit_y(df_labels, lineprice_cols)

    # --- Collect sequences ---
    X_dicts_list, y_list, x_lengths, label_lengths = [], [], [], []
    feature_cols, feature_columns = None, {}

    for _, row in df_labels.iterrows():
        mask = (
            (feature_pipeline.main_data["timestamp"] >= row["startTime"])
            & (feature_pipeline.main_data["timestamp"] <= row["endTime"])
        )
        df_main = feature_pipeline.main_data.loc[mask].copy()

        subseqs = {"main": df_main.copy()}
        if feature_pipeline is not None:
            for k, v in feature_pipeline.global_dicts.items():
                subseqs[k] = v.loc[mask].copy()

            subseqs = feature_pipeline.apply_window(subseqs)
            if subseqs is None:
                continue
            subseqs = feature_pipeline._normalize(subseqs, fit=False)

        X_dict = {}
        for dict_name, df_sub in subseqs.items():
            feats = [c for c in df_sub.columns if c != "timestamp"]
            if dict_name not in feature_columns:
                feature_columns[dict_name] = feats.copy()
            if dict_name == "main" and feature_cols is None:
                feature_cols = feats

            arr = df_sub[feats].values.astype(np.float32)
            X_dict[dict_name] = arr

        if not X_dict or arr.shape[0] == 0:
            continue

        X_dicts_list.append(X_dict)
        x_lengths.append(len(subseqs["main"]))

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

    # --- Window normalization across sequences ---
    if feature_pipeline is not None and getattr(feature_pipeline, "window_norms", None):
        X_dicts_list = feature_pipeline.apply_window_normalization(
            X_dicts_list, feature_columns, fit=window_norm_fit
        )

    # --- Pad labels ---
    max_len_y = max((len(arr) for arr in y_list), default=0)
    y = np.zeros((len(y_list), max_len_y), dtype=np.float32)
    for i, arr in enumerate(y_list):
        y[i, : len(arr)] = arr

    # ===============================
    # Return in requested format
    # ===============================
    if for_xgboost:
        from utils.padd_sequence_xgboost import pad_sequences_dicts
        X_main = pad_sequences_dicts(X_dicts_list, dict_name="main", strategy="forward_fill")

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

        if val_split:
            idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_state)
            return (
                X_main[idx_train], y[idx_train],
                X_main[idx_val],   y[idx_val],
                df_labels, feature_columns, max_len_y
            )
        else:
            return X_main, y, df_labels, feature_columns, max_len_y

    # Torch Dataset mode
    dataset = build_multiinput_dataset(X_dicts_list, y, x_lengths)

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
        X_train = [X_dicts_list[i] for i in idx_train]
        X_val   = [X_dicts_list[i] for i in idx_val]
        return (
            build_multiinput_dataset(X_train, y[idx_train], [x_lengths[i] for i in idx_train]),
            build_multiinput_dataset(X_val,   y[idx_val],   [x_lengths[i] for i in idx_val]),
            df_labels, feature_columns, max_len_y
        )
    else:
        return dataset, df_labels, feature_columns, max_len_y


