from flask import Flask, render_template, jsonify, request, make_response
import pandas as pd
import io
import csv
from tpo_logic import get_tpo_levels

app = Flask(__name__)

# Load Data Once
# Note: Adjust format if your CSV is strictly DD/MM or MM/DD. 
# dayfirst=True handles 1/1/2018 as Jan 1st correctly for international formats.
df = pd.read_csv('/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=False) # dayfirst=False implies MDY often, set True if DMY
df = df.sort_values('timestamp')

# Store calculated results in memory
calculated_results = []

@app.route('/')
def index():
    return render_template('tpo.html')

@app.route('/data')
def get_data():
    """Returns chart data formatted for Lightweight Charts"""
    chart_data = []
    for _, row in df.iterrows():
        chart_data.append({
            # Lightweight charts expects UNIX timestamp (seconds)
            'time': int(row['timestamp'].timestamp()), 
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })
    return jsonify(chart_data)

@app.route('/calculate_profile', methods=['POST'])
def calculate_profile():
    """Receives start/end timestamps, slices DF, runs TPO logic"""
    req = request.json
    start_ts = req.get('start') # Unix timestamp
    end_ts = req.get('end')     # Unix timestamp

    # Convert unix to datetime
    start_dt = pd.to_datetime(start_ts, unit='s')
    end_dt = pd.to_datetime(end_ts, unit='s')

    # Filter DataFrame
    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
    window_df = df.loc[mask].copy()

    if window_df.empty:
        return jsonify({'error': 'No data in range'}), 400

    levels = get_tpo_levels(window_df)
    
    if levels:
        # Add timestamps for record keeping
        result_entry = {
            "startTime": str(start_dt),
            "endTime": str(end_dt),
            "VAL": levels['VAL'],
            "VAH": levels['VAH'],
            "POC": levels['POC']
        }
        calculated_results.append(result_entry)
        return jsonify(levels)
    else:
        return jsonify({'error': 'Calculation failed'}), 500

@app.route('/download_csv')
def download_csv():
    """Downloads the accumulated results"""
    si = io.StringIO()
    cw = csv.writer(si)
    
    # Header
    cw.writerow(['startTime', 'endTime', 'VAL', 'VAH', 'POC'])
    
    # Rows
    for res in calculated_results:
        cw.writerow([res['startTime'], res['endTime'], res['VAL'], res['VAH'], res['POC']])
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=tpo_results.csv"
    output.headers["Content-type"] = "text/csv"
    return output

if __name__ == '__main__':
    app.run(debug=True, port=5000)