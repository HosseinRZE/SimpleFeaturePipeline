# train_transformer.py
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset, random_split
from preprocess import load_raw_data_serve, prepare_sequences
from models.transformer.transform_classifier import TransformerClassifier
import pytorch_lightning as pl

SEQ_LEN = 3

df = load_raw_data_serve("/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
                         "/home/iatell/projects/meta-learning/data/labeled_ohlcv_string.csv")
X, y, label_encoder = prepare_sequences(df, seq_len=SEQ_LEN)

dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = TransformerClassifier(
    input_dim=X.shape[2],
    seq_len=SEQ_LEN,
    d_model=64,
    num_heads=4,
    num_layers=2,
    num_classes=len(label_encoder.classes_),
    lr=1e-3
)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

joblib.dump({
    "seq_len": SEQ_LEN,
    "input_dim": X.shape[2],
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 2,
    "num_classes": len(label_encoder.classes_),
    "lr": 1e-3,
    "label_encoder": label_encoder
}, "models/saved_models/transformer_meta.pkl")

trainer.save_checkpoint("models/saved_models/transformer.ckpt")
