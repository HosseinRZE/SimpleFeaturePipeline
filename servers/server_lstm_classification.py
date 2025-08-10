import glob
import joblib
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from preprocess import load_raw_data_serve, one_sample
from models.LSTM.lstm_classifier import LSTMClassifier

app = Flask(__name__)

# ---------- Load meta & model once at start-up ----------
meta_path = glob.glob("models/saved_models/lstm_meta_*.pkl")[0]
state_path = glob.glob("models/saved_models/lstm_*.ckpt")[0]

meta = joblib.load(meta_path)

model = LSTMClassifier(
    input_dim=meta['input_dim'],
    hidden_dim=meta['hidden_dim'],
    num_layers=meta['num_layers'],
    num_classes=meta['num_classes'],
    lr=meta['lr']
)
model.load_state_dict(torch.load(state_path, map_location='cpu')["state_dict"])
model.eval()

# ---------- Load candle data ----------
df = load_raw_data_serve(
    "/path/to/full_candles.csv",
    "/path/to/labeled_dataset.csv"
)

# Keep track of chart index
class State:
    idx = meta['seq_len'] - 1
state = State()

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("chart_lstm.html")

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

@app.route("/predict", methods=['POST'])
def predict():
    idx = request.json['idx']  # right-most candle index
    seq_df = df.iloc[idx - meta['seq_len'] + 1: idx + 1]
    X_np = one_sample(seq_df)  # shape (seq_len, feature_dim)
    X_t = torch.from_numpy(X_np.astype(np.float32)).unsqueeze(0)  # (1, seq_len, feat)

    with torch.no_grad():
        logits = model(X_t)
        pred_class = torch.argmax(logits, dim=1).item()

    return jsonify({
        'class': int(pred_class),
        'candle_time': int(seq_df.iloc[-1]['timestamp'].timestamp())
    })

if __name__ == "__main__":
    app.run(debug=True)
