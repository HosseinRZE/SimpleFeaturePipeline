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
from utils.multilabel_threshold_tuning import tune_thresholds_nn

def evaluate_model(model, val_loader, mlb, threshold=0.2, return_probs=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    print("\n📊 Validation Report (Multi-label):")
    print(classification_report(all_labels, all_preds, target_names=mlb.classes_, zero_division=0))

    print("\n🧮 Multi-label Confusion Matrices (per class):")
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    for i, cls in enumerate(mlb.classes_):
        print(f"\nClass '{cls}':")
        print(mcm[i])

    val_acc_exact = np.all(all_preds == all_labels, axis=1).mean()
    val_acc_micro = (all_preds == all_labels).mean()
    print("\nExact match ratio:", val_acc_exact)
    print("Micro accuracy (per-label):", val_acc_micro)

    if return_probs:
        return val_acc_exact, val_acc_micro, all_probs
    else:
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
        train_ds, val_ds, label_encoder, df, feature_cols, label_weights = preprocess_csv_multilabel(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=True,
            debug_sample=True,
            label_weighting="none"  # or "none", or {"i": 2}
        )
    else:
        full_dataset, label_encoder, df, feature_cols, label_weights = preprocess_csv_multilabel(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=False,
            debug_sample=True,
            label_weighting="none"
        )   

    # --- Model config ---
    input_dim = train_ds[0][0].shape[1] if do_validation else full_dataset[0][0].shape[1]
    num_classes = len(label_encoder.classes_)
    label_weights_tensor = torch.tensor(label_weights, dtype=torch.float32)

    model = LSTMMultiLabelClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        lr=lr,
        label_weights_tensor=label_weights_tensor
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
        # --- Step 1: Evaluate with default threshold ---
        val_acc_exact_default, val_acc_micro_default, y_probs = evaluate_model(
            model, val_loader, label_encoder, threshold=0.5, return_probs=True
        )

        # --- Step 2: Tune thresholds per label ---
        optimal_thresholds = tune_thresholds_nn(
            y_true=np.vstack([y for _, y in val_loader.dataset]),
            y_probs=y_probs
        )
        print("\n📌 Optimal thresholds per label:", dict(zip(label_encoder.classes_, optimal_thresholds)))

        # --- Step 3: Evaluate again with tuned thresholds ---
        val_acc_exact_tuned, val_acc_micro_tuned, _ = evaluate_model(
            model, val_loader, label_encoder, threshold=0.5, return_probs=True
        )
        # Apply per-label thresholds manually
        y_pred_tuned = (y_probs >= np.array(optimal_thresholds)).astype(int)
        val_acc_exact_tuned = np.all(y_pred_tuned == np.vstack([y for _, y in val_loader.dataset]), axis=1).mean()
        val_acc_micro_tuned = (y_pred_tuned == np.vstack([y for _, y in val_loader.dataset])).mean()

        print(f"\n✅ Validation before tuning: Exact={val_acc_exact_default:.3f}, Micro={val_acc_micro_default:.3f}")
        print(f"✅ Validation after tuning: Exact={val_acc_exact_tuned:.3f}, Micro={val_acc_micro_tuned:.3f}")


if __name__ == "__main__":
    train_model(
        data_csv="data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        labels_json="data/candle_labels.json",  # JSON labels, no CSV needed on disk
        do_validation=True,
        label_weighting="scale_pos"
    )
