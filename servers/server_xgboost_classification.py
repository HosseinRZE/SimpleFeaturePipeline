import glob
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from xgboost import XGBClassifier

from preprocess import compute_candle_features  # reuse from earlier code

# -------------------
# Flask app
# -------------------
app = Flask(__name__)

# -------------------
# 1. Load model + encoder at startup
# -------------------
MODEL_DIR = "models/saved_models"
model_path = glob.glob(os.path.join(MODEL_DIR, "xgboost_candle_model.json"))[0]
encoder_path = glob.glob(os.path.join(MODEL_DIR, "label_encoder.pkl"))[0]

model = XGBClassifier()
model.load_model(model_path)
label_encoder = joblib.load(encoder_path)

# -------------------
# 2. Load candle CSV
# -------------------
CANDLE_CSV = "/path/to/your/full_ohlc_dataset.csv"  # should have timestamp, open, high, low, close
df = pd.read_csv(CANDLE_CSV)

# Ensure timestamp is datetime
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Compute features
df = compute_candle_features(df)

# -------------------
# 3. State to track current position
# -------------------
class State:
    n_candles = 3  # model trained on last 3 candles
state = State()

# -------------------
# ROUTES
# -------------------
@app.route("/")
def home():
    return render_template("chart.html")  # your chart HTML

@app.route("/candles")
def candles():
    """Send OHLC candles to browser."""
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

@app.route("/predict", methods=["POST"])
def predict():
    """Predict class for the candle at index `idx`."""
    idx = request.json['idx']  # right-most candle on screen

    if idx < state.n_candles:
        return jsonify({'error': 'Not enough candles for prediction'}), 400

    # Select last n_candles before idx
    feature_cols = ['open','high','low','close',
                    'upper_shadow','body','lower_shadow',
                    'upper_body_ratio','lower_body_ratio','Candle_Color']

    past_candles = df.iloc[idx - state.n_candles: idx][feature_cols].values.flatten().reshape(1, -1)

    # Predict
    proba = model.predict_proba(past_candles)[0]
    pred_idx = np.argmax(proba)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return jsonify({
        'class': str(pred_label),
        'probabilities': {label: float(prob) for label, prob in zip(label_encoder.classes_, proba)}
    })

if __name__ == '__main__':
    app.run(debug=True)
