# server.py
import json, joblib, glob, numpy as np, pandas as pd
from flask import Flask, request, jsonify, render_template
import preprocess.xgboost_data_p as p

app = Flask(__name__)

# load once at start-up
model_p  = glob.glob("models/saved_models/model_p*.pkl")[0]
model_mu = glob.glob("models/saved_models/model_mu*.pt")[0]
df       = p.load_data( "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
                       "/home/iatell/projects/meta-learning/data/ohlcv_log(2).csv")

class State:
    idx = p.SEQ_LEN - 1
state = State()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/candles")
def candles():
    data = []
    for _, row in df.iterrows():
        data.append({
            'time': int(row['timestamp'].timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low':  float(row['low']),
            'close':float(row['close'])
        })
    return jsonify(data)

import traceback

@app.route("/predict", methods=['POST'])
def predict():
    try:
        idx   = request.json['idx']
        seq   = df.iloc[idx - p.SEQ_LEN + 1 : idx + 1]  # 5 rows
        x_vec = p.one_feature_vector(seq).reshape(1, -1)

        p_hat = model_p.predict(x_vec)[0]
        mu    = np.exp(model_mu.predict(x_vec)[0])

        return jsonify({'line': bool(p_hat >= 0.5),
                        'price': float(seq.iloc[-1]['close'] * mu)})
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(ex)}), 500

if __name__ == '__main__':
    app.run(debug=True)