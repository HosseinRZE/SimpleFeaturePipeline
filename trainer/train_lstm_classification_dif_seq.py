import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from preprocess.classification_dif_seq import preprocess_sequences_csv
from models.LSTM.lstm_dif_seq_class import LSTMClassifier
from utils.print_batch import print_batch
from utils.to_address import to_address
from utils.json_to_csv import json_to_csv_in_memory
from utils.padding_batch import collate_batch
import pandas as pd 
import io
# ---------------- Evaluation ---------------- #
def evaluate_model(model, val_loader, label_encoder):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch, lengths in val_loader:
            logits = model(X_batch, lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print("\n📊 Validation Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)


# ---------------- Train ---------------- #
def train_model(
    data_csv,
    labels_csv,
    model_out_dir="models/saved_models",
    do_validation=True,
    hidden_dim=128,
    num_layers=1,
    lr=0.001,
    batch_size=32,
    max_epochs=50,
    save_model=True,
    return_val_accuracy=True,
    test_mode=False
):
    """
    Train an LSTM classifier with variable-length sequences.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/lstm_model_seq_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/lstm_meta_seq_{timestamp}.pkl"

    # --- Get dataset(s) --- #
    if do_validation:
        train_ds, val_ds, label_encoder, df, feature_cols = preprocess_sequences_csv(
            data_csv, labels_csv, val_split=True
        )
    else:
        full_dataset, label_encoder, df, feature_cols = preprocess_sequences_csv(
            data_csv, labels_csv, val_split=False
        )

    # --- Model config --- #
    input_dim = train_ds[0][0].shape[1] if do_validation else full_dataset[0][0].shape[1]
    num_classes = len(label_encoder.classes_)

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        lr=lr
    )

    # --- DataLoaders --- #
    if do_validation:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_batch)
    else:
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader   = None

    # Debug batch
    if test_mode:
        global df_seq
        df_seq = print_batch(train_loader, feature_cols, batch_idx=2)

    # --- Trainer --- #
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        fast_dev_run=test_mode,
    )

    trainer.fit(model, train_loader, val_loader)

    # --- Save --- #
    if save_model:
        trainer.save_checkpoint(model_out)
        joblib.dump({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'lr': lr,
            'label_classes': label_encoder.classes_
        }, meta_out)
        print(f"\n✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # --- Evaluation --- #
    val_acc = None
    if do_validation:
        evaluate_model(model, val_loader, label_encoder)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch, lengths in val_loader:
                logits = model(X_batch, lengths)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        val_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()

    if return_val_accuracy:
        return {"accuracy": val_acc}


if __name__ == "__main__":
    train_model(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        to_address(pd.read_csv(io.StringIO(json_to_csv_in_memory("/home/iatell/projects/meta-learning/data/string_sequence.json")))),
        do_validation=True
    )
