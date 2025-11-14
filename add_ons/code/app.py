from flask import Flask, render_template, request, jsonify
import pandas as pd
import io
from datetime import datetime
import json

app = Flask(__name__)
global_ohlc_data = []

@app.route('/')
def index():
    return render_template('real.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    # NEW: Access the global variable to modify it
    global global_ohlc_data 
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        csv_content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Validate required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400
        
        # Convert timestamp to datetime and then to Unix timestamp
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y')
        except:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                return jsonify({'error': 'Invalid timestamp format. Expected MM/DD/YYYY'}), 400
        
        df['time'] = df['timestamp'].astype('int64') // 10**9  # Convert to seconds
        
        # Validate numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                return jsonify({'error': f'Invalid numeric data in {col} column'}), 400
        
        # Sort by timestamp
        df = df.sort_values('time')
        
        # Prepare data for frontend
        ohlc_data = []
        volume_data = []
        
        for _, row in df.iterrows():
            ohlc_data.append({
                'time': int(row['time']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
            
            # Determine volume color based on price movement
            color = 'rgba(0, 150, 136, 0.7)' if row['close'] > row['open'] else 'rgba(255, 82, 82, 0.7)'
            volume_data.append({
                'time': int(row['time']),
                'value': float(row['volume']),
                'color': color
            })
        
        # NEW: Store the complete OHLC data in our global variable
        global_ohlc_data = ohlc_data
        
        return jsonify({
            'success': True,
            'data': {
                'ohlc': ohlc_data,
                'volume': volume_data
            }
        })
        
    except Exception as e:
        # NEW: Clear global data on error
        global_ohlc_data = [] 
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/calculate_profile', methods=['POST'])
def calculate_profile():
    # NEW: Access the global variable to read from it
    global global_ohlc_data
    
    try:
        data = request.json
        # NEW: Rename the variable to show it's just the range boundaries
        boundary_candles = data.get('candles', []) 
        
        if len(boundary_candles) < 2:
            return jsonify({'error': 'At least 2 candles required'}), 400
            
        # NEW: Check if the global data is available
        if not global_ohlc_data:
            return jsonify({'error': 'No data uploaded. Please upload a CSV file first.'}), 400

        # NEW: Get the start and end times from the two candles provided
        start_time = boundary_candles[0]['time']
        end_time = boundary_candles[-1]['time']

        # NEW: Filter the *global* data to get ALL candles in the range
        candles = [
            candle for candle in global_ohlc_data
            if start_time <= candle['time'] <= end_time
        ]

        # NEW: Validate the *filtered* list
        if len(candles) < 2:
            return jsonify({'error': 'Not enough candle data found in the selected time range.'}), 400
        
        # --- FROM HERE, THE REST OF THE ALGORITHM WORKS AS-IS ---
        # It is now using the 'candles' variable, which contains the *full* set
        # of candles in the range, not just the two boundaries.
        
        # 1. Extract all OHLC prices from all candles
        # 2. Sort them ascending
        # 3. Create buckets between consecutive prices
        # 4. Count how many times each bucket is touched by any candle's range
        
        all_prices = []
        for candle in candles:
            all_prices.extend([candle['open'], candle['high'], candle['low'], candle['close']])
        
        # Remove duplicates and sort
        unique_prices = sorted(list(set(all_prices)))
        
        if len(unique_prices) < 2:
            return jsonify({'error': 'Not enough unique price points'}), 400
        
        # Create buckets between consecutive prices
        buckets = []
        for i in range(len(unique_prices) - 1):
            buckets.append({
                'start': unique_prices[i],
                'end': unique_prices[i + 1],
                'volume': 0,
                'touched_by': []  # Track which candles touch this bucket
            })
        
        # Count how many times each bucket is touched by candle ranges
        maxVolume = 0
        pocBucket = None
        
        for i, candle in enumerate(candles):
            candle_low = min(candle['open'], candle['high'], candle['low'], candle['close'])
            candle_high = max(candle['open'], candle['high'], candle['low'], candle['close'])
            
            for bucket in buckets:
                # Check if candle's range overlaps with this bucket
                if candle_low <= bucket['end'] and candle_high >= bucket['start']:
                    bucket['volume'] += 1
                    bucket['touched_by'].append(i)  # Track which candle touched this bucket
        
        # Find POC (bucket with highest volume)
        for bucket in buckets:
            if bucket['volume'] > maxVolume:
                maxVolume = bucket['volume']
                pocBucket = bucket
        
        # Mark POC bucket
        for bucket in buckets:
            bucket['isPoc'] = bucket == pocBucket
            # Remove the touched_by array as it's not needed for frontend
            bucket.pop('touched_by', None)
        
        # Reverse for display (high to low)
        buckets.reverse()
        
        return jsonify({
            'success': True,
            'buckets': buckets,
            'maxVolume': maxVolume,
            'totalCandles': len(candles) # This now correctly reports all candles
        })
        
    except Exception as e:
        return jsonify({'error': f'Error calculating profile: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)