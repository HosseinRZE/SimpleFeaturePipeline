# train_lstm.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocess import make_datasets, SEQ_LEN
from models.LSTM.lstm_classifier import LSTMClassifier
import joblib

def train_model(csv_path, model_out="lstm_model.pt", meta_out="lstm_meta.pkl"):
    train_ds, test_ds, df = make_datasets(csv_path)
    input_dim = train_ds[0][0].shape[1]
    num_classes = len(set(df['label']))
    hidden_dim = 64
    num_layers = 1
    lr = 0.001

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        lr=lr
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(test_ds, batch_size=32)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1 if pl.accelerators.is_available("gpu") else None,
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)

    # Save model weights
    trainer.save_checkpoint(model_out)

    # Save meta info
    joblib.dump({
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'seq_len': SEQ_LEN,
        'lr': lr
    }, meta_out)

if __name__ == "__main__":
    train_model("candles_with_labels.csv")
