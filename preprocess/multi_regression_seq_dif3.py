import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from sklearn.model_selection import train_test_split
from utils.build_multi_input_dataset import build_multiinput_dataset
from utils.debug_samples import debug_print_samples


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
    max_len_y = max(label_lengths, default=0)
    y = np.zeros((len(y_list), max_len_y), dtype=np.float32)
    for i, arr in enumerate(y_list):
        nonzero = arr[arr != 0]  # keep only real labels
        y[i, : len(nonzero)] = nonzero

    # ===============================
    # XGBoost mode for Multi-regression
    # ===============================
    if for_xgboost:
        if feature_pipeline is None or feature_pipeline.transformations is None:
            raise ValueError("XGBoost mode requires feature_pipeline with transformations defined")

        # Apply feature transformations
        X_trans = feature_pipeline.apply_transformations(X_dicts_list)

        # Debug before splitting
        if debug_sample:
            indices = [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
            debug_print_samples(indices, X_trans=X_trans, y_arr=y)

        if val_split:
            # Use train_test_split to divide into train and validation sets
            (
                X_train, X_val,
                y_train, y_val,
                idx_train, idx_val
            ) = train_test_split(
                X_trans, y, np.arange(len(y)),
                test_size=test_size, random_state=random_state
            )

            # also split lengths
            train_lengths_x = [x_lengths[i] for i in idx_train]
            val_lengths_x   = [x_lengths[i] for i in idx_val]
            train_lengths_y = [label_lengths[i] for i in idx_train]
            val_lengths_y   = [label_lengths[i] for i in idx_val]

            out = (
                X_train, y_train, X_val, y_val,
                feature_columns, max_len_y,
                (train_lengths_x, train_lengths_y),
                (val_lengths_x, val_lengths_y),
            )

        else:
            # If no validation split, return the entire dataset
            out = (
                X_trans, y,
                feature_columns, max_len_y,
                (x_lengths, label_lengths),
            )

        return out


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


