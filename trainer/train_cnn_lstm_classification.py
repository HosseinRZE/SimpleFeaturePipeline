import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from utils.print_batch import print_batch

from preprocess.classification_pre_dict import preprocess_csv
from models.LSTM.cnn_lstm_classifier import CNNLSTMClassifier

def evaluate_model(model, val_loader, label_encoder):
    """Generate classification report & confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []

    device = next(model.parameters()).device
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch.to(device))
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
    return_val_accuracy=False,
    cnn_channels=(64,),
    cnn_kernel_sizes=(3,),
    cnn_strides=(1,),
    cnn_paddings=(0,),
    test_mode = False
):
    # --- Timestamp for unique filenames ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/cnn_lstm_class_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/cnn_lstm_meta_{timestamp}.pkl"

    # --- Get dataset(s) ---
    if do_validation:
        train_ds, val_ds, label_encoder, df, feature_cols = preprocess_csv(
            data_csv, labels_csv, n_candles=seq_len, val_split=True
        )
    else:
        full_dataset, label_encoder, df, feature_cols = preprocess_csv(
            data_csv, labels_csv, n_candles=seq_len, val_split=False
        )

    if test_mode:
        global df_seq
        df_seq = print_batch(train_loader, feature_cols, batch_idx=2)
    # --- Model config ---
    input_dim = train_ds[0][0].shape[1] if do_validation else full_dataset[0][0].shape[1]
    num_classes = len(label_encoder.classes_)

    model = CNNLSTMClassifier(
        input_dim=input_dim,
        seq_len= seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        lr=lr,
        cnn_channels=cnn_channels,
        cnn_kernel_sizes=cnn_kernel_sizes,
        cnn_strides=cnn_strides,
        cnn_paddings=cnn_paddings
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
        devices=1 if torch.cuda.is_available() else "auto",
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)

    # --- Save model ---
    if save_model:
        trainer.save_checkpoint(model_out)
        joblib.dump({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'seq_len': seq_len,
            'lr': lr,
            'label_classes': label_encoder.classes_,
            'cnn_channels': cnn_channels,
            'cnn_kernel_sizes': cnn_kernel_sizes,
            'cnn_strides': cnn_strides,
            'cnn_paddings': cnn_paddings
        }, meta_out)
        print(f"\n✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # --- Optional evaluation ---
    val_acc = None
    if do_validation:
        device = next(model.parameters()).device
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(device))
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        val_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)

        evaluate_model(model, val_loader, label_encoder)

    if return_val_accuracy:
        return {"accuracy": val_acc if val_acc is not None else 0.0}


if __name__ == "__main__":
    train_model(
        "data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        "data/labeled_ohlcv_string.csv",
        do_validation=True
    )
