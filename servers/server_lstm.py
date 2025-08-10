# server.py  (relevant lines only)
import glob, joblib, torch, numpy as np
from models.LSTM.simple_lstm import LSTMLinePredictor
from preprocess import sequential as dp         # <- renamed predictor → data_pipeline
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
# 1. load meta & model once at start-up
meta_path = glob.glob("models/saved_models/lstm_meta_*.pkl")[0]
state_path = glob.glob("models/saved_modles/lstm_*.pt")[0]

meta   = joblib.load(meta_path)
model  = LSTMLinePredictor(feature_dim=meta['input_dim'])
model.load_state_dict(torch.load(state_path, map_location='cpu'))
model.eval()

# 2. load candle/label CSV
df = dp.load_raw_data_serve(
    "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
    "/home/iatell/projects/meta-learning/data/ohlcv_log(2).csv"
)

class State:
    idx = meta['seq_len'] - 1
state = State()

@app.route("/")
def home():
    return render_template("sequential.html")

@app.route("/candles")
def candles():
    return jsonify([
        {'time': int(r['timestamp'].timestamp()),
         'open': float(r['open']),
         'high': float(r['high']),
         'low':  float(r['low']),
         'close':float(r['close'])}
        for _, r in df.iterrows()
    ])

@app.route("/predict", methods=['POST'])
def predict():
    idx = request.json['idx']          # idx is the **right-most candle** on screen
    seq_df = df.iloc[idx - meta['seq_len'] + 1: idx + 1]
    X_np   = dp.one_sample(seq_df)     # (seq_len, feature_dim)
    X_t    = torch.from_numpy(X_np.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        p_pred, mu_pred = model(X_t)
    return jsonify({
        'prob':  float(p_pred[0].item()),
        'price': float(seq_df.iloc[-1]['close'] * np.exp(mu_pred[0].item()))
    })

if __name__ == '__main__':
    app.run(debug=True)