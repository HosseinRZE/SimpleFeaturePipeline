# train_transformer.py
import os, datetime, joblib, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from lightning import Trainer
from preprocess import sequential as p
from models.transformer.simple_transformer import TransformerPredictor

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

model = TransformerPredictor(feature_dim,SEQ_LEN)
trainer = Trainer(max_epochs=20, log_every_n_steps=10)
trainer.fit(model, loader)
# 4. save
ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("models", "saved_models")
os.makedirs(save_dir, exist_ok=True)
torch.save(
    model.state_dict(),
    os.path.join(save_dir, f"transformer{ts}.pt")
)
joblib.dump(
    {"seq_len": SEQ_LEN, "input_dim": feature_dim},
    os.path.join(save_dir, f"transformer_meta_{ts}.pkl")
)
print("Transformer saved → models/")