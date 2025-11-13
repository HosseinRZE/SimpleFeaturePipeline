# 2-Candle Volume Profile Analyzer

A web application for analyzing custom volume profiles between two selected candles using a unique bucketing method.

## Features

- 📊 Interactive candlestick charts using TradingView Lightweight Charts
- 📁 CSV data upload with automatic parsing
- 🎯 Custom volume profile calculation between any two candles
- 🎨 Visual representation of price buckets with POC (Point of Control) highlighting
- 📱 Responsive design for desktop and mobile
- 🔄 Real-time profile calculation and display

## How It Works

The application uses a unique volume profile calculation method:

1. **Data Input**: Upload a CSV file with OHLCV data (timestamp, open, high, low, close, volume)
2. **Candle Selection**: Click any two candles on the chart to analyze
3. **Price Extraction**: All OHLC prices from selected candles are extracted and sorted
4. **Bucket Creation**: Price ranges are created between consecutive sorted prices
5. **Touch Counting**: Each bucket is counted based on how many candle ranges overlap with it
6. **Profile Display**: Shows touch frequency distribution with the POC (most touched bucket) highlighted

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## CSV File Format

Your CSV file should have the following columns:
- `timestamp`: Date in MM/DD/YYYY format
- `open`: Opening price
- `high`: High price
- `low`: Low price  
- `close`: Closing price
- `volume`: Volume data

Example:
```
timestamp,open,high,low,close,volume
1/1/2018,13707.91,13818.55,12750,13380,8607.1564
1/2/2018,13382.16,15473.49,12890.02,14675.11,20078.1654
```

## Usage

1. **Upload Data**: Click "Choose File" and select your CSV file
2. **View Chart**: The candlestick chart will load with your data
3. **Select Candles**: Click on any two candles to analyze
   - First click: Selects the first candle (blue marker)
   - Second click: Selects the second candle (orange marker)
4. **View Profile**: The volume profile appears on the right panel showing:
   - Price buckets with volume levels
   - POC (Point of Control) in gold
   - Visual bars representing volume distribution

## Technical Details

### Backend (Flask)
- RESTful API endpoints for file upload and profile calculation
- CSV parsing and validation
- JSON data exchange with frontend

### Frontend
- TradingView Lightweight Charts for visualization
- Responsive design with CSS Grid and Flexbox
- Asynchronous data loading with Fetch API
- Interactive candle selection with visual markers

### Volume Profile Algorithm
```
1. Extract all OHLC prices from selected candles
2. Sort all prices in ascending order
3. Create buckets between consecutive sorted prices
4. For each bucket:
   - Count how many candle ranges overlap with this bucket
5. Identify POC (bucket with highest touch count)
6. Display profile with buckets sorted high-to-low
```

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## License

This project is open source and available under the MIT License.

## Troubleshooting

### Common Issues

1. **CSV Upload Fails**
   - Check file format matches required structure
   - Ensure all numeric columns contain valid numbers
   - Verify timestamp format is MM/DD/YYYY

2. **Chart Not Loading**
   - Check browser console for JavaScript errors
   - Ensure all dependencies are installed
   - Verify Flask server is running on port 5000

3. **Profile Not Calculating**
   - Make sure exactly two different candles are selected
   - Check that candles have valid OHLC data
   - Verify network connection for API calls

### Support

For issues and feature requests, please create an issue in the project repository.