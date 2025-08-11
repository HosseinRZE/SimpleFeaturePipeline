from flask import Flask, render_template, jsonify
import pandas as pd
import time
from utils.zigzag import ZigZag  # Use the cleaned class from earlier

app = Flask(__name__)

# Load data
price_data = pd.read_csv("/home/iatell/financial_data/BTC15min.csv")
price_data.rename(columns={
    'column6': 'volume',
    'column5': 'close',
    'column4': 'low',
    'column3': 'high',
    'column2': 'open'
}, inplace=True)

# State
current_index = 0
candles = []

@app.route("/")
def index():
    return render_template("zigzag.html")

@app.route("/data")
def get_data():
    global current_index, candles
    if current_index < len(price_data):
        row = price_data.iloc[current_index]
        # Use a consistent timestamp
        ts = int(time.time()) + current_index * 60  # or row['timestamp'] if available
        candles.append({
            "time": ts,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close
        })
        current_index += 1

    # Recalculate ZigZag
    df = pd.DataFrame(candles)
    zigzag_points = []
    if len(df) >= 3:
        zz = ZigZag(df, window_size=3, dev_threshold=1, shadow_mode=True)
        pivots = zz.give_zigzag()
        zigzag_points = [
            {
                "time": c["time"],  # Match candle time!
                "value": p.price
            }
            for p in pivots
            for c in candles if c["open"] == p.price or c["high"] == p.price or c["low"] == p.price or c["close"] == p.price
        ]

    return jsonify({
        "candles": candles,
        "zigzag": zigzag_points
    })
if __name__ == "__main__":
    app.run(debug=True)
