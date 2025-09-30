import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from utils.debug_samples import debug_print_samples
from utils.build_multi_input_dataset import build_multiinput_dataset


def preprocess_sequences_csv_multilabels(
    data_csv,
    labels_csv,
    feature_pipeline=None,
    val_split=True,
    test_size=0.2,
    random_state=42,
    for_xgboost=False,
    debug_sample=False,
    window_norm_fit=True,
    return_mlb=True,
):
    """
    Preprocess sequences for multi-label classification.
    """

    # --- Load data ---
    df_data = pd.read_csv(data_csv)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])

    df_labels = pd.read_csv(labels_csv)
    df_labels['startTime'] = pd.to_datetime(df_labels['startTime'], unit='s')
    df_labels['endTime']   = pd.to_datetime(df_labels['endTime'], unit='s')

    # --- Fit pipeline globally ---
    if feature_pipeline is not None:
        feature_pipeline.fit(df_data)

    # --- Prepare label encoder ---
    def _parse_label_field(x):
        if pd.isna(x):
            return []
        return [p.strip() for p in str(x).split(",") if p.strip() != ""]

    all_label_lists = df_labels['labels'].apply(_parse_label_field).tolist()
    mlb = MultiLabelBinarizer()
    mlb.fit(all_label_lists if any(all_label_lists) else [[]])
    num_classes = len(mlb.classes_)

    # --- Collect sequences ---
    X_dicts_list, y_list, x_lengths, label_counts = [], [], [], []
    feature_cols = None
    feature_columns = {}

    for _, row in df_labels.iterrows():
        mask = (
            (feature_pipeline.main_data['timestamp'] >= row['startTime']) &
            (feature_pipeline.main_data['timestamp'] <= row['endTime'])
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
        arr = None
        for dict_name, df_sub in subseqs.items():
            feats = [c for c in df_sub.columns if c != "timestamp"]
            if dict_name not in feature_columns:
                feature_columns[dict_name] = feats.copy()
            if dict_name == "main" and feature_cols is None:
                feature_cols = feats
            arr = df_sub[feats].values.astype(np.float32)
            X_dict[dict_name] = arr

        if not X_dict or arr is None or arr.shape[0] == 0:
            continue

        X_dicts_list.append(X_dict)
        x_lengths.append(len(subseqs["main"]))

        labels = _parse_label_field(row.get('labels', []))
        y_bin = mlb.transform([labels])[0] if num_classes > 0 else np.zeros((num_classes,), dtype=np.float32)
        y_list.append(y_bin.astype(np.float32))
        label_counts.append(len(labels))

    # --- Window normalization ---
    if feature_pipeline is not None and getattr(feature_pipeline, "window_norms", None):
        X_dicts_list = feature_pipeline.apply_window_normalization(
            X_dicts_list, feature_columns, fit=window_norm_fit
        )

    if len(y_list) == 0:
        # no samples collected
        empty_y = np.zeros((0, num_classes), dtype=np.float32)
        if for_xgboost:
            out = (np.zeros((0, 0)), empty_y, df_labels, feature_columns, 0, [])
        else:
            out = (build_multiinput_dataset([], empty_y, []), df_labels, feature_columns, 0)
        return out + (mlb,) if return_mlb else out

    y_arr = np.vstack(y_list).astype(np.float32)
    max_y = int(max(label_counts)) if label_counts else 0
    label_counts_arr = np.array(label_counts, dtype=int)

    # ===============================
    # XGBoost mode
    # ===============================
    if for_xgboost:
        if feature_pipeline is None or feature_pipeline.transformations is None:
            raise ValueError("XGBoost mode requires feature_pipeline with transformations defined")
        
        X_trans = feature_pipeline.apply_transformations(X_dicts_list)

        # Debug before splitting
        if debug_sample:
            indices = [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
            debug_print_samples(indices, X_trans=X_trans, y_arr=y_arr, mlb=mlb)

        if val_split:
            X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
                X_trans, y_arr, np.arange(len(y_arr)), test_size=test_size, random_state=random_state
            )
            train_counts = label_counts_arr[idx_train]
            val_counts = label_counts_arr[idx_val]
            out = (X_train, y_train, X_val, y_val, df_labels, feature_columns, max_y, train_counts, val_counts)
        else:
            out = (X_trans, y_arr, df_labels, feature_columns, max_y, label_counts_arr)

        return out + (mlb,) if return_mlb else out

    # ===============================
    # Torch Dataset mode
    # ===============================
    dataset = build_multiinput_dataset(X_dicts_list, y_arr, x_lengths)

    if val_split:
        idx_train, idx_val = train_test_split(np.arange(len(y_arr)), test_size=test_size, random_state=random_state)
        train_dataset = build_multiinput_dataset([X_dicts_list[i] for i in idx_train], y_arr[idx_train], [x_lengths[i] for i in idx_train])
        val_dataset   = build_multiinput_dataset([X_dicts_list[i] for i in idx_val],   y_arr[idx_val],   [x_lengths[i] for i in idx_val])
        out = (train_dataset, val_dataset, df_labels, feature_columns, max_y)
    else:
        out = (dataset, df_labels, feature_columns, max_y)

    if debug_sample:
        indices = [debug_sample] if isinstance(debug_sample, int) else list(debug_sample)
        debug_print_samples(indices, X_dicts_list=X_dicts_list, y_arr=y_arr, mlb=mlb)

    return out + (mlb,) if return_mlb else out
