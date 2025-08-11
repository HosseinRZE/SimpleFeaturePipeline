
from flask import Flask, render_template, jsonify
import pandas as pd
import time
from utils.zigzag_reverse_incremental import ZigZag  # Use the cleaned class from earlier
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
# Initialize ZigZag
zz = ZigZag(window_size=3, dev_threshold=1, shadow_mode=True)

# Reset with empty dataframe but correct columns
zz.reset(pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"]))
# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('zigzag_reverse.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # prevent 404 spam in logs

@app.route("/data")
def get_data():
    global current_index, candles
    if current_index < len(price_data):
        row = price_data.iloc[current_index]
        candle = {
            "time": int(row['timestamp']),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "timestamp": int(row['timestamp'])
        }
        candles.append(candle)

        # Update ZigZag with only the new candle
        zz.update_with_new_candle(candle)

        current_index += 1

    return jsonify({
        "candles": candles,
        "zigzag": zz.get_pivots()
    })
if __name__ == "__main__":
    app.run(debug=True)
