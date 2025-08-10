# train.py
import os, joblib, numpy as np
from xgboost import XGBRegressor
import preprocess.xgboost_data_p as p
import datetime
from pathlib import Path
ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
CANDLE  = "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv"
LABEL   = "/home/iatell/projects/meta-learning/data/ohlcv_log(2).csv"
MODEL_P  = f"models/model_p_{ts}.pkl"
MODEL_MU = f"models/model_mu_{ts}.pkl"
df = p.load_data(CANDLE, LABEL)
# -------------------------------------------------
# 1. load and align
X_all, y_cls_all, y_reg_all = p.make_features(df)
# ensure 2-D
assert X_all.ndim == 2, f"X_all is {X_all.ndim}D"
# 2. regression subset (rows that have a label)
mask = ~np.isnan(y_reg_all)
X_reg   = X_all[mask]        # already aligned
y_reg   = y_reg_all[mask]
# balanced classification
neg = np.where(y_cls_all == 0)[0]
pos = np.where(y_cls_all == 1)[0]
neg_keep = np.random.choice(neg, size=int(0.1 * len(neg)), replace=False)
idx_cls = np.concatenate([pos, neg_keep])

X_cls_bal = X_all[idx_cls]
y_cls_bal = y_cls_all[idx_cls]

idx_pos = idx_cls[y_cls_bal == 1]
pos_in_reg = np.searchsorted(np.where(y_cls_all == 1)[0], idx_pos)
X_reg_bal = X_reg[pos_in_reg]
y_reg_bal = y_reg[pos_in_reg]

model_p  = XGBRegressor(objective='binary:logistic', n_estimators=400,
                        max_depth=4, learning_rate=0.03, subsample=0.9)
model_mu = XGBRegressor(objective='reg:squarederror', n_estimators=400,
                        max_depth=4, learning_rate=0.03, subsample=0.9)

save_dir = Path("models/saved_models")
save_dir.mkdir(parents=True, exist_ok=True)

MODEL_P = save_dir / "model_p.pkl"
MODEL_MU = save_dir / "model_mu.pkl"

model_p.fit(X_cls_bal, y_cls_bal)
model_mu.fit(X_reg_bal, y_reg_bal)

joblib.dump(model_p, MODEL_P)
joblib.dump(model_mu, MODEL_MU)

print(f"Models saved → {save_dir}")