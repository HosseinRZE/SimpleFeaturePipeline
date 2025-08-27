import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from itertools import islice
from preprocess.classification_pre_dict import preprocess_csv
from models.LSTM.lstm_classifier import LSTMClassifier
import pandas as pd 
from utils.print_batch import print_batch


def evaluate_model(model, val_loader, label_encoder):
    """
    Evaluate a trained model on validation data.

    Args:
        model (torch.nn.Module): Trained LSTM classifier.
        val_loader (DataLoader): DataLoader for validation dataset.
        label_encoder (LabelEncoder): Fitted label encoder (for readable class names).

    Prints:
        - Classification report (precision, recall, f1, support).
        - Confusion matrix.
    """
    model.eval()  # switch to evaluation mode
    all_preds, all_labels = [], []

    with torch.no_grad():  # disable gradient tracking for speed
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)               # forward pass
            preds = torch.argmax(logits, dim=1)   # predicted class index
            all_preds.extend(preds.cpu().numpy()) # move to CPU, store
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
    hidden_dim=10,
    num_layers=1,
    lr=0.001,
    batch_size=32,
    max_epochs=50,
    save_model=True,
    return_val_accuracy=True,
    test_mode = False
):
    """
    Train an LSTM classification model using PyTorch Lightning.

    Args:
        data_csv (str): Path to candles CSV file (OHLCV data).
        labels_csv (str): Path to labels CSV file (class labels).
        model_out_dir (str, optional): Directory where model & metadata are saved.
        do_validation (bool, optional): If True, split data into train/val sets.
        seq_len (int, optional): Number of candles per sequence (LSTM input length).
        hidden_dim (int, optional): Size of hidden state in LSTM.
        num_layers (int, optional): Number of stacked LSTM layers.
        lr (float, optional): Learning rate for optimizer.
        batch_size (int, optional): Batch size for DataLoader.
        max_epochs (int, optional): Number of training epochs.
        save_model (bool, optional): If True, save model checkpoint & metadata.
        return_val_accuracy (bool, optional): If True, return validation accuracy.

    Returns:
        dict | None: {"accuracy": float} if return_val_accuracy=True, else None.

        Notes:
        - If `do_validation=True`, this function calls `preprocess_csv` with `val_split=True`
          and receives:
            • train_ds (TensorDataset): training set, each element is (X_i, y_i).
                - X_i shape: (seq_len, feature_dim), e.g. (3, 10) for one sequence.
                - y_i: integer class label for that sequence.
                - All data combined has shape:
                    X_train.shape = (num_train_samples, seq_len, feature_dim)
                    y_train.shape = (num_train_samples,)
            • val_ds (TensorDataset): validation set, same format as train_ds.
            • label_encoder (LabelEncoder): maps original string labels → integer classes.
            • df (DataFrame): merged OHLCV data + labels for reference/inspection.

        - If `do_validation=False`, it receives:
            • full_dataset (TensorDataset): entire dataset without split.
            • label_encoder (LabelEncoder).
            • df (DataFrame).

        - `input_dim` is automatically inferred from the dataset:
            • It is the number of features per candle (columns in FEATURE_COLS).
            • Computed as: `train_ds[0][0].shape[1]` if validation is enabled,
              otherwise from `full_dataset`.
            • For example, with FEATURE_COLS = 10, input_dim = 10.

        - These datasets are wrapped into DataLoaders so PyTorch Lightning can feed
          `(X_batch, y_batch)` pairs into the model during training.    
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/lstm_model_class_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/lstm_meta_class_{timestamp}.pkl"

    # --- Get dataset(s) ---
    # --- Get dataset(s) ---
    if do_validation:
        train_ds, val_ds, label_encoder, df, feature_cols = preprocess_csv(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=True,
            # feature_pipeline=pipeline
        )
    else:
        full_dataset, label_encoder, df, feature_cols = preprocess_csv(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=False,
            # feature_pipeline=pipeline
        )

    # --- Model config ---
    # Determine input dimension (#features per time step)
    input_dim = train_ds[0][0].shape[1] if do_validation else full_dataset[0][0].shape[1]
    num_classes = len(label_encoder.classes_)

    # Initialize model
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

    # --- print a sample
    # --- Debug: Inspect one batch being fed to LSTM ---
    if test_mode:
        global df_seq
        df_seq = print_batch(train_loader, feature_cols, batch_idx=2)

    # --- Trainer setup ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",   # automatically picks "gpu" if available, else "cpu"
        devices=1,            # use 1 device (GPU if available)
        log_every_n_steps=10,
        fast_dev_run=test_mode,    # ✅ runs 1 batch for train + 1 batch for val, no full training
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # --- Save model & metadata ---
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
    # Example: training with validation split
    train_model(
        "data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        "data/labeled_ohlcv_string.csv",
        do_validation=True
    )
