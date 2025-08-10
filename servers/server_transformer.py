# server_attention.py
import glob, joblib, torch, numpy as np
from flask import Flask, request, jsonify, render_template
from models.transformer.simple_transformer import TransformerPredictor
from preprocess import sequential as dp

app = Flask(__name__)

# 1. load meta & attention model once
meta_path  = glob.glob("models/saved_models/transformer_meta_*.pkl")[0]
state_path = glob.glob("models/saved_models/transformer_*.pt")[0]

meta  = joblib.load(meta_path)
model = TransformerPredictor(feature_dim=meta['input_dim'], seq_len=meta['seq_len'])
model.load_state_dict(torch.load(state_path, map_location='cpu'))
model.eval()

# 2. same daily-dense data
df = dp.load_raw_data_serve(
    "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
    "/home/iatell/projects/meta-learning/data/ohlcv_log(2).csv"
)

@app.route("/")
def home():
    return render_template("sequential.html")

@app.route("/candles")
def candles():
    dense = df.set_index('timestamp').asfreq('D').ffill()
    return jsonify([
        {'time': int(ts.timestamp()),
         'open': float(row.open),
         'high': float(row.high),
         'low':  float(row.low),
         'close':float(row.close)}
        for ts, row in dense.iterrows()
    ])

@app.route("/predict", methods=['POST'])
def predict():
    idx   = request.json['idx']
    seq_df = df.iloc[idx - meta['seq_len'] + 1 : idx + 1]
    X_np   = dp.one_sample(seq_df)
    X_t    = torch.from_numpy(X_np.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        p_raw, mu_raw = model(X_t)

    return jsonify({
        'prob':  float(p_raw[0].item()),
        'price': float(seq_df.iloc[-1]['close'] * np.exp(mu_raw[0].item()))
    })

if __name__ == '__main__':
    app.run(debug=True)