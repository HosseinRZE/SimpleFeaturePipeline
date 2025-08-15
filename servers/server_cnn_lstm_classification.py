import glob
import joblib
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from preprocess.classification_pre import load_raw_data_serve, one_sample
from models.LSTM.cnn_lstm_classifier import CNNLSTMClassifier

app = Flask(__name__)

# ---------- Load meta & model once at start-up ----------
meta_path = sorted(glob.glob("models/saved_models/cnn_lstm_meta_*.pkl"))[-1]
state_path = sorted(glob.glob("models/saved_models/cnn_lstm_class_*.pt"))[-1]

meta = joblib.load(meta_path)

model = CNNLSTMClassifier(
    input_dim=meta['input_dim'],
    seq_len=meta['seq_len'],
    hidden_dim=meta['hidden_dim'],
    num_layers=meta['num_layers'],
    num_classes=meta['num_classes'],
    lr=meta['lr'],
    cnn_channels=meta['cnn_channels'],
    cnn_kernel_sizes=meta['cnn_kernel_sizes'],
    cnn_strides=meta['cnn_strides'],
    cnn_paddings=meta['cnn_paddings']
)
model.load_state_dict(torch.load(state_path, map_location='cpu')["state_dict"])
model.eval()

# ---------- Load candle data ----------
df = load_raw_data_serve(
    "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
    "/home/iatell/projects/meta-learning/data/labeled_ohlcv_string.csv"
)

# Keep track of chart index
class State:
    idx = meta['seq_len'] - 1
state = State()

# ---------- Routes ----------
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
            'error': f"Not enough candles to make prediction (need {meta['seq_len']}, got {idx+1})",
            'class': None,
            'logits': None,
            'candle_time': None
        }), 400

    seq_df = df.iloc[idx - meta['seq_len'] + 1: idx + 1]
    X_np = one_sample(seq_df)
    X_t = torch.from_numpy(X_np.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits_t = model(X_t)
        pred_class = torch.argmax(logits_t, dim=1).item()
        logits_list = logits_t.squeeze(0).tolist()

    return jsonify({
        'class': int(pred_class),
        'logits': logits_list,
        'candle_time': int(seq_df.iloc[-1]['timestamp'].timestamp())
    })


if __name__ == "__main__":
    app.run(debug=True)
