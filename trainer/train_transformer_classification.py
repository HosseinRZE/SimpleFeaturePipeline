import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from utils.print_batch import print_batch
from preprocess.classification_pre_dict import preprocess_csv
from models.transformer.transform_classifier  import TransformerClassifier  # <-- your Transformer

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
    do_validation=True,
    seq_len=3,
    d_model=64,
    num_heads=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    lr=0.001,
    batch_size=32,
    max_epochs=10,
    save_model=False,
    return_val_accuracy=True,
    test_mode = True
):
    """
    Train a Transformer classification model.

    Args:
        data_csv (str): Path to candles CSV.
        labels_csv (str): Path to labels CSV.
        model_out_dir (str): Folder to save model & metadata.
        do_validation (bool): Whether to split data for validation.
        seq_len (int): Number of candles per sequence.
        d_model (int): Transformer embedding size.
        num_heads (int): Attention heads.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int): Feedforward layer size.
        dropout (float): Dropout rate.
        lr (float): Learning rate.
        batch_size (int): Training batch size.
        max_epochs (int): Max training epochs.
        save_model (bool): Save model & meta to disk.
        return_val_accuracy (bool): If True, return validation accuracy.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/transformer_model_class_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/transformer_meta_class_{timestamp}.pkl"

    # --- Get dataset(s) ---
    if do_validation:
        train_ds, val_ds, label_encoder, df, feature_cols = preprocess_csv(
            data_csv, labels_csv, n_candles=seq_len, val_split=True
        )
    else:
        full_dataset, label_encoder, df, feature_cols = preprocess_csv(
            data_csv, labels_csv, n_candles=seq_len, val_split=False
        )

    # --- Model config ---
    input_dim = train_ds[0][0].shape[1] if do_validation else full_dataset[0][0].shape[1]
    num_classes = len(label_encoder.classes_)

    model = TransformerClassifier(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
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

    if test_mode:
        global df_seq
        df_seq = print_batch(train_loader, feature_cols, batch_idx=2)
    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,  # always at least 1 device
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    # --- Save model & meta ---
    if save_model:
        trainer.save_checkpoint(model_out)
        joblib.dump({
            'input_dim': input_dim,
            'seq_len': seq_len,
            'd_model': d_model,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'num_classes': num_classes,
            'lr': lr,
            'label_classes': label_encoder.classes_
        }, meta_out)
        print(f"\n✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # --- Optional evaluation ---
    val_acc = None
    if do_validation:
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
        "data/labeled_ohlcv_candle.csv",
        do_validation=True
    )