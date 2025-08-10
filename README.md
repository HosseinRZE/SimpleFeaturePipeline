```markdown
# 📈  BTC Horizontal-Line Predictors  
*CNN-LSTM, LSTM, Transformer, XGBoost – same dataset, same chart*

---

## 🎯  Goal
Predict the **coefficient** (≈ 0.99 … 1.01) of horizontal **support / resistance** lines on daily BTC candles.  
Each model outputs  
- `p`  – probability that a line exists  
- `μ`  – log-coefficient (final price = last-close × exp(μ))

---

## 🧱  Project Layout
```
project/
├── data/
│   ├── Bitcoin_BTCUSDT_1D_candles_prop.csv   # OHLCV + engineered features
│   └── ohlcv_log(2).csv                      # labels: timestamp,last_close,(line)
├── models/                                   # *.pt / *.pkl created by trainers
├── predictor.py                              # shared preprocessing
├── train_<model>.py                          # one script per model
├── server_<model>.py                         # Flask server with /predict endpoint
└── templates/index.html                      # Lightweight-Charts front-end
```

---

## 🔧  One-time Setup
```bash
# 1. create virtual env
python -m venv venv && source venv/bin/activate

# 2. install core stack
pip install torch lightning flask pandas numpy joblib xgboost
```

---

## 🚀  Usage (repeat for every model)

| Model      | Train once                          | Serve forever                       |
|------------|-------------------------------------|-------------------------------------|
| **XGBoost**| `python train.py`                   | `python server.py`                  |
| **LSTM**   | `python train_lstm.py`              | `python server_lstm.py`             |
| **CNN-LSTM**| `python train_cnn_lstm.py`          | `python server_cnn_lstm.py`         |
| **Transformer**| `python train_transformer.py`   | `python server_transformer.py`      |

Open **http://localhost:5000** and click **NEXT CANDLE** to watch the chart update with the model’s predictions.

---

## 📊  Data Pipeline (predictor.py)
- **Window sizes**: 1-, 3-, 5-candle sequences  
- **Normalisation**: OHLC divided by **last-close**  
- **Meta features**: upper_shadow, body, lower_shadow, upper_body_ratio, lower_body_ratio  

---

## 🧪  Training Notes
- Labels are extracted from parentheses: `(1.003)` → 1.003, `(-1)` → no line  
- Positive class (line) down-sampled to 10 % to keep balance  
- Models saved with UTC timestamp: `lstm_20240805_143215.pt`

---

## 🌐  API
| Endpoint       | Method | Body               | Response                               |
|----------------|--------|--------------------|----------------------------------------|
| `/candles`     | GET    | —                  | Array of `{time,open,high,low,close}` |
| `/predict`     | POST   | `{"idx": 123}`     | `{"line": true/false, "price": 12345}` |

---

## 📄  License
MIT – feel free to fork & extend.
```