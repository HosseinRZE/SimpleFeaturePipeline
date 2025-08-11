
from flask import Flask, render_template, jsonify
import pandas as pd
import time
from utils.zigzag_reverse import ZigZag  # Use the cleaned class from earlier
from flask import Flask, render_template, jsonify

 # Assuming your ZigZag class is in zigzag.py

app = Flask(__name__)

# ---------- Load and prepare dataset ----------
df = pd.read_csv("/home/iatell/financial_data/BTC15min.csv")

df.rename(columns={
    'column6': 'volume',
    'column5': 'close',
    'column4': 'low',
    'column3': 'high',
    'column2': 'open',
    'column1': 'timestamp'
}, inplace=True)

# Detect milliseconds and convert
if df['timestamp'].iloc[0] > 1e12:
    df['timestamp'] = df['timestamp'] // 1000

# Create datetime index
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('datetime', inplace=True)

# Final OHLC dataset with timestamp for chart
price_data = df[['open', 'high', 'low', 'close', 'volume', 'timestamp']]

# Reverse for backward stepping
price_data = price_data[::-1].reset_index(drop=True)

# ---------- Globals ----------
current_index = 0
candles = []

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('zigzag_reverse.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # prevent 404 spam in logs

@app.route('/data')
def get_data():
    global current_index, candles

    # Add one candle per request
    if current_index < len(price_data):
        row = price_data.iloc[current_index]
        candles.append({
            "time": int(row['timestamp']),  # UNIX seconds
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close)
        })
        current_index += 1

    if not candles:
        return jsonify({"candles": [], "zigzag": []})

    # Recalculate ZigZag only if enough data
    zigzag_points = []
    if len(candles) >= 3:
        df_candles = pd.DataFrame(candles)
        zz = ZigZag(df_candles, window_size=3, dev_threshold=3, shadow_mode=True)
        pivots = zz.give_zigzag(backward=True)
        print("*************")
        # Ensure pivot times match actual candles
        for p in pivots:
            print("pivot price:",p.price)    
            if 0 <= p.index < len(df_candles):
                zigzag_points.append({
                    "time": int(df_candles.iloc[p.index]["time"]),
                    "value": float(p.price)
                })
    return jsonify({
        "candles": candles,
        "zigzag": zigzag_points
    })

if __name__ == "__main__":
    app.run(debug=True)
