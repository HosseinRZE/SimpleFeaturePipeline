# app.py
from flask import Flask, render_template, request
import pandas as pd
import json
from utils.zigzag_bandf import ZigZag

app = Flask(__name__)

# ---------- Load CSV data once ----------
DATA_CSV = "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv"
LABELS_CSV = "data/labeled_ohlcv_string.csv"

df_raw = pd.read_csv(DATA_CSV)
df_labels = pd.read_csv(LABELS_CSV).rename(columns={"labels": "label"})
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
df_labels["timestamp"] = pd.to_datetime(df_labels["timestamp"])
df_merged = pd.merge(df_raw, df_labels[["timestamp", "label"]], on="timestamp", how="left")


def prepare_chart_data(row_index, window_size=100):
    global df_merged  

    zz = ZigZag(
        window_size=3,
        dev_threshold=1,
        max_pivots=10,
        include_last_candle_as_pivot=True,
        stationary=False,
        include_distances=True
    )

    features_list = []
    for idx, row in df_merged.iloc[:row_index+1].iterrows():
        zz.update(idx, row["close"])
        # Pass current index for proper distance calculation
        features = zz.get_features(row["close"], idx)
        features_list.append(features)

    df_feat = df_merged.iloc[:row_index+1].copy().reset_index(drop=True)
    feat_df = pd.DataFrame(features_list)
    df_feat = pd.concat([df_feat, feat_df], axis=1)

    # Candles for plotting (window)
    start = max(0, row_index - window_size // 2)
    end = min(len(df_feat), row_index + window_size // 2 + 1)
    df_window = df_feat.iloc[start:end].reset_index(drop=True)

    candles = [
        {
            "time": int(ts.timestamp()),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c)
        }
        for ts, o, h, l, c in zip(
            df_window["timestamp"],
            df_window["open"],
            df_window["high"],
            df_window["low"],
            df_window["close"]
        )
    ]

# Feed candles one by one
    for idx, row in df_merged.iloc[:row_index+1].iterrows():
        zz.update(idx, row["close"])  # or row["high"], row["low"], row["close"] if you want

    # Get drawable pivots
    pivots = zz.draw(df=df_merged, max_index=row_index)

    return candles, pivots, df_feat.iloc[-1].to_dict(), len(df_merged)


@app.route("/", methods=["GET"])
def index():
    default_idx = len(df_merged) // 2
    row_index = int(request.args.get("row_index", default_idx))
    row_index = max(0, min(len(df_merged) - 1, row_index))

    candles, pivot_line, row_data, max_index = prepare_chart_data(row_index)

    return render_template(
        "chart.html",
        row_index=row_index,
        max_index=len(df_merged) - 1,
        candles_json=json.dumps(candles),
        pivot_line_json=json.dumps(pivot_line),
        row_data=row_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
