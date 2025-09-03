import numpy as np
import pandas as pd
def add_label_normalized_candles(df, labels_csv, ohlc_cols=["open","high","low","close"]):
    labels_df = pd.read_csv(labels_csv)

    # Drop rows with missing indices
    labels_df = labels_df.dropna(subset=["startIndex", "endIndex"])

    # Force integer conversion safely
    labels_df["startIndex"] = labels_df["startIndex"].astype(float).astype(int)
    labels_df["endIndex"]   = labels_df["endIndex"].astype(float).astype(int)
    max_len = (labels_df["endIndex"] - labels_df["startIndex"] + 1).max()
    new_cols = []

    for i in range(max_len):
        for col in ohlc_cols:
            new_cols.append(f"{col}_{i}")

    flat_data = []
    for _, row in labels_df.iterrows():
        start_idx = int(row["startIndex"])
        end_idx   = int(row["endIndex"])

        window = df.iloc[start_idx:end_idx+1][ohlc_cols].values.astype(np.float32)
        last_close = window[-1, 3]
        norm_window = window / last_close

        pad_len = max_len - norm_window.shape[0]
        if pad_len > 0:
            pad = np.tile(norm_window[0], (pad_len, 1))
            norm_window = np.vstack([pad, norm_window])
        flat_data.append(norm_window.flatten())

    df_new = pd.DataFrame(flat_data, columns=new_cols)
    df_combined = pd.concat([df.reset_index(drop=True), df_new], axis=1)
    return df_combined
