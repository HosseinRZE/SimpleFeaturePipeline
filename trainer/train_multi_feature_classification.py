import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from add_ons.zigzag_single import add_zigzag
from preprocess.classification_pre2 import preprocess_csv
from models.LSTM.lstm_classifier import LSTMClassifier

def evaluate_model(model, val_loader, label_encoder):
    """Generate classification report & confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print("\n📊 Validation Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)


def train_model(
    data_csv,
    labels_csv,
    model_out_dir="models/saved_models",
    do_validation=False,
    seq_len=3,
    hidden_dim=64,
    num_layers=1,
    lr=0.001,
    batch_size=32,
    max_epochs=10,
    save_model=True,
    return_val_accuracy=False
):
    """
    Train an LSTM classification model with zigzag features.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/lstm_model_class_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/lstm_meta_class_{timestamp}.pkl"

    # 🔹 Define feature pipeline (zigzag for now, more can be added)
    feature_pipeline = [
        lambda df: add_zigzag(df, window_size=3, dev_threshold=1, max_pivots=10)
    ]

    # --- Get dataset(s) ---
    if do_validation:
        train_ds, val_ds, label_encoder, df, feature_cols= preprocess_csv(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=True,
            feature_pipeline=feature_pipeline
        )
    else:
        full_dataset, label_encoder, df, feature_cols = preprocess_csv(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=False,
            feature_pipeline=feature_pipeline
        )

    # --- Model config ---
    input_dim = train_ds[0][0].shape[1] if do_validation else full_dataset[0][0].shape[1]
    num_classes = len(label_encoder.classes_)

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        lr=lr
    )

    # --- DataLoaders ---
    if do_validation:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size)
    else:
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = None

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    # --- Save model & meta ---
    if save_model:
        trainer.save_checkpoint(model_out)
        joblib.dump({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'seq_len': seq_len,
            'lr': lr,
            'label_classes': label_encoder.classes_
        }, meta_out)
        print(f"\n✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # --- Optional evaluation ---
    val_acc = None
    if do_validation and val_loader is not None:
        evaluate_model(model, val_loader, label_encoder)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        val_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()

    if return_val_accuracy:
        return {"accuracy": val_acc}


if __name__ == "__main__":
    train_model(
        "data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        "data/labeled_ohlcv_string.csv",
        do_validation=True
    )
