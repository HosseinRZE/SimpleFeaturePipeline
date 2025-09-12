from flask import Flask, render_template, jsonify
import pandas as pd
from add_ons.candle_dif_rate_of_change_percentage2 import add_candle_rocp
from add_ons.candle_proportion_simple import add_candle_shape_features
app = Flask(__name__)

# Load CSV
df = pd.read_csv("/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_features, invalid_idx = add_candle_rocp(df)
df_features, invalid_idx_shape = add_candle_shape_features(df_features)
# Prepare OHLC data for chart
ohlc_data = df_features[['timestamp','open','high','low','close']].copy()

# Convert datetime to string if you have string timestamps
ohlc_data['time'] = ohlc_data['timestamp'].dt.strftime('%Y-%m-%d')

# OR if you prefer numbers (Unix timestamp in seconds)
# ohlc_data['time'] = ohlc_data['timestamp'].astype(int) // 10**9

ohlc_data = ohlc_data.drop(columns=['timestamp'])

@app.route("/")
def index():
    return render_template("ohlc_features.html")

@app.route("/ohlc_data")
def get_ohlc_data():
    data = ohlc_data.to_dict(orient='records')
    return jsonify(data)

@app.route("/candle_features/<int:index>")
def get_candle_features(index):
    if index < 0 or index >= len(df_features):
        return jsonify({"error": "Invalid index"}), 400
    row = df_features.iloc[index].to_dict()
    return jsonify(row)

if __name__ == "__main__":
    app.run(debug=True)
