import os, datetime, joblib, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from lightning import Trainer
from models.LSTM.simple_lstm import LSTMLinePredictor
from preprocess import sequential as p
"""
Constants & Pipeline
--------------------
SEQ_LEN      = 3
    How many consecutive daily bars constitute one training sample.

feature_dim  = len(p.FEATURES)   # 10
    Number of features per daily bar (OHLC + engineered meta).

Data loading
------------
df = p.load_raw_data(CANDLE_CSV, LABEL_CSV)
    Reads raw CSV files, merges candles with labels, and returns a cleaned
    DataFrame that contains every candle plus two target columns:
        - has_line   : binary 0/1 flag
        - log_factor : log-price label (NaN when no label)

Sequence creation
-----------------
X_seq, y_cls, y_reg = p.build_sequences(df)
    Converts the cleaned DataFrame into tensors:
        X_seq : (N, SEQ_LEN, feature_dim)  — 3-day windows of features
        y_cls : (N,)                       — binary label for each window
        y_reg : (N,)                       — log-price label for each window
    Rows with NaN labels are already dropped inside build_sequences.

Balanced classification subset
------------------------------
mask_cls = y_cls.astype(bool) | (np.random.rand(len(y_cls)) < 0.1)
    Keeps every positive sample (has_line == 1) and keeps ~10 % of the
    negatives so the classifier sees roughly balanced classes.

X_cls, y_cls_bal = X_seq[mask_cls], y_cls[mask_cls]
    Balanced feature/label tensors for the classification head.

Regression subset (no re-balancing)
-----------------------------------
mask_reg = ~np.isnan(y_reg)
    Identifies windows that have a valid log-price label.

X_reg, y_reg_bal = X_seq[mask_reg], y_reg[mask_reg]
    Features and labels for the regression head.

Unified dataset (recommended)
-----------------------------
X_all, y_cls_all, y_reg_all = p.build_sequences(df)
    Same as X_seq, y_cls, y_reg — already NaN-cleaned.

train_ds = TensorDataset(
    torch.from_numpy(X_all).float(),
    torch.from_numpy(y_cls_all.astype(np.float32)),
    torch.from_numpy(y_reg_all.astype(np.float32))
)
    Single PyTorch dataset returning triplets
    (features, binary_label, regression_label).

loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    Randomly shuffled, batched loader feeding the Lightning trainer.
"""
CANDLE_CSV = "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv"
LABEL_CSV  = "/home/iatell/projects/meta-learning/data/ohlcv_log(2).csv"
SEQ_LEN    = 3
feature_dim = len(p.FEATURES)        # 10

df = p.load_raw_data(CANDLE_CSV, LABEL_CSV)

X_all, y_cls_all, y_reg_all = p.build_sequences(df)   # already filtered together

n_total   = len(y_cls_all)
print(f"Total windows : {n_total}")

train_ds = TensorDataset(
    torch.from_numpy(X_all).float(),
    torch.from_numpy(y_cls_all.astype(np.float32)),
    torch.from_numpy(y_reg_all.astype(np.float32))
)
loader = DataLoader(train_ds, batch_size=128, shuffle=True)

model = LSTMLinePredictor(feature_dim)
trainer = Trainer(max_epochs=20, log_every_n_steps=10)
trainer.fit(model, loader)

ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("models", "saved_models")
os.makedirs(save_dir, exist_ok=True)
torch.save(
    model.state_dict(),
    os.path.join(save_dir, f"lstm{ts}.pt")
)
joblib.dump(
    {"seq_len": SEQ_LEN, "input_dim": feature_dim},
    os.path.join(save_dir, f"lstm_meta_{ts}.pkl")
)
print(f"lstm saved → {save_dir}")