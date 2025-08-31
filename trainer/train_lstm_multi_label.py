import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from datetime import datetime
import pandas as pd
import io
import os
import numpy as np
from preprocess.multilabel_preprocess import preprocess_csv_multilabel
from models.LSTM.lstm_multi_label import LSTMMultiLabelClassifier
from utils.print_batch import print_batch
from utils.json_to_csv import json_to_csv_in_memory  # <-- new util


def evaluate_model(model, val_loader, mlb, threshold=0.2):
    """
    Evaluate a trained multi-label LSTM model on a validation dataset.

    This function computes predictions from the model, converts them into
    binary labels using a specified threshold, and reports several metrics
    to assess performance in a multi-label classification setting.

    Args:
        model (torch.nn.Module): The trained LSTM multi-label classifier.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        mlb (sklearn.preprocessing.MultiLabelBinarizer): Fitted label binarizer
            used to encode the labels.
        threshold (float, optional): Probability threshold to convert sigmoid
            outputs to binary labels (default is 0.2).

    Prints:
        - Multi-label classification report (precision, recall, F1-score) per label.
        - Multi-label confusion matrix for each label.
        - Exact match ratio: fraction of samples where all labels are predicted correctly.
        - Micro accuracy: fraction of individual label predictions that are correct across all samples.

    Returns:
        tuple:
            val_acc_exact (float): Exact match ratio across all samples.
            val_acc_micro (float): Micro accuracy per label.

    Notes:
        - Exact match ratio is stricter than micro accuracy; it requires all labels
          of a sample to be predicted correctly.
        - Micro accuracy gives a single number summarizing the overall label-wise
          prediction accuracy, useful for comparing models across datasets.
        - The classification report uses `zero_division=0` to avoid undefined metrics
          for labels with no predictions or no true samples.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            probs = torch.sigmoid(logits)          # multi-label probabilities
            preds = (probs >= threshold).int()     # convert to binary 0/1
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    print("\n📊 Validation Report (Multi-label):")
    print(classification_report(all_labels, all_preds, target_names=mlb.classes_, zero_division=0))

    print("\n🧮 Multi-label Confusion Matrices (per class):")
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    for i, cls in enumerate(mlb.classes_):
        print(f"\nClass '{cls}':")
        print(mcm[i])

    # exact match ratio across all samples
    val_acc_exact = np.all(all_preds == all_labels, axis=1).mean()
    print("\nExact match ratio:", val_acc_exact)

    # micro accuracy per label
    val_acc_micro = (all_preds == all_labels).mean()
    print("Micro accuracy (per-label):", val_acc_micro)

    return val_acc_exact, val_acc_micro


def train_model(
    data_csv,
    labels_json=None,
    model_out_dir="models/saved_models",
    do_validation=True,
    seq_len=1,
    hidden_dim=10,
    num_layers=1,
    lr=0.001,
    batch_size=32,
    max_epochs=200,
    save_model=False,
    return_val_accuracy=True,
    test_mode=False,
):
    """
    Train an LSTM classification model with labels coming from JSON (in-memory CSV).
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/lstm_model_class_{timestamp}.pt"
    meta_out = f"{model_out_dir}/lstm_meta_class_{timestamp}.pkl"

    # --- Prepare labels ---
    if labels_json is not None:
        csv_string = json_to_csv_in_memory(labels_json)   # returns CSV string
        labels_csv = io.StringIO(csv_string)              # file-like for pandas
    else:
        raise ValueError("labels_json must be provided")

    # --- Get dataset(s) ---
    if do_validation:
        train_ds, val_ds, label_encoder, df, feature_cols = preprocess_csv_multilabel(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=True,debug_sample=True
        )
    else:
        full_dataset, label_encoder, df, feature_cols = preprocess_csv_multilabel(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=False,debug_sample=True
        )

    # --- Model config ---
    input_dim = train_ds[0][0].shape[1] if do_validation else full_dataset[0][0].shape[1]
    num_classes = len(label_encoder.classes_)

    model = LSTMMultiLabelClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        lr=lr,
    )

    # --- DataLoaders ---
    if do_validation:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
    else:
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        val_loader = None

    # --- Debug batch ---
    if test_mode:
        global df_seq
        df_seq = print_batch(train_loader, feature_cols, batch_idx=2)

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        fast_dev_run=test_mode,
    )

    trainer.fit(model, train_loader, val_loader)

    # --- Save model & metadata ---
    if save_model:
        os.makedirs(model_out_dir, exist_ok=True)
        trainer.save_checkpoint(model_out)
        joblib.dump({
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "seq_len": seq_len,
            "lr": lr,
            "label_classes": label_encoder.classes_,
        }, meta_out)
        print(f"\n✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # --- Validation accuracy ---
    val_acc_exact, val_acc_micro = None, None
    if do_validation:
        val_acc_exact, val_acc_micro = evaluate_model(model, val_loader, label_encoder)


if __name__ == "__main__":
    train_model(
        data_csv="data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        labels_json="data/candle_labels.json",  # JSON labels, no CSV needed on disk
        do_validation=True,
    )
