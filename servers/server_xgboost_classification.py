import glob
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from preprocess.classification_pre import load_raw_data_serve, one_sample

app = Flask(__name__)

meta_path = glob.glob("models/saved_models/xgb_meta_class*.pkl")[0]
model_path = glob.glob("models/saved_models/xgb_model_class*.pkl")[0]

meta = joblib.load(meta_path)
model = joblib.load(model_path)

df = load_raw_data_serve(
    "data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
    "data/labeled_ohlcv_string.csv"
)

@app.route("/")
def home():
    return render_template("classification.html")

@app.route("/candles")
def candles():
    return jsonify([
        {
            'time': int(r['timestamp'].timestamp()),
            'open': float(r['open']),
            'high': float(r['high']),
            'low':  float(r['low']),
            'close':float(r['close'])
        }
        for _, r in df.iterrows()
    ])

@app.route("/meta")
def meta_info():
    return jsonify({
        'label_classes': list(meta['label_classes'])
    })

@app.route("/predict", methods=['POST'])
def predict():
    idx = request.json['idx']

    if idx < meta['seq_len'] - 1:
        return jsonify({
            'error': f"Not enough candles (need {meta['seq_len']}, got {idx+1})",
            'class': None,
            'logits': None,
            'candle_time': None
        }), 400

    seq_df = df.iloc[idx - meta['seq_len'] + 1: idx + 1]
    X_np = one_sample(seq_df).flatten().reshape(1, -1)

    pred_class = model.predict(X_np)[0]
    logits_list = model.predict_proba(X_np)[0].tolist()

    return jsonify({
        'class': int(pred_class),
        'logits': logits_list,
        'candle_time': int(seq_df.iloc[-1]['timestamp'].timestamp())
    })

if __name__ == "__main__":
    app.run(debug=True)
