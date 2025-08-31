import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from preprocess.multi_regression_seq_dif import preprocess_sequences_csv_multilines
from models.LSTM.lstm_multi_line_reg_seq_dif import MultiLineLSTMRegressor
from utils.print_batch import print_batch
from utils.to_address import to_address
from utils.json_to_csv import json_to_csv_in_memory
from utils.padding_batch import collate_batch
import pandas as pd
import io
import numpy as np

# ---------------- Evaluation ---------------- #
def evaluate_model(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch, lengths in val_loader:
            preds = model(X_batch, lengths)   # regression outputs
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    mse = ((all_preds - all_labels) ** 2).mean()
    mae = np.abs(all_preds - all_labels).mean()

    print("\n📊 Validation Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    return mse, mae


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
    save_model=False,
    return_val_accuracy=True,
    test_mode=False,
    n_candles=None,
    feature_pipeline=None
):
    """
    Train an LSTM regressor with variable-length multi-line sequences.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/lstm_model_multiline_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/lstm_meta_multiline_{timestamp}.pkl"

    # --- Get dataset(s) --- #
    if do_validation:
        train_ds, val_ds, df, feature_cols = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=True,
            n_candles=n_candles,
            feature_pipeline=feature_pipeline
        )
    else:
        full_dataset, df, feature_cols = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=False,
            n_candles=n_candles,
            feature_pipeline=feature_pipeline
        )

    # --- Model config --- #
    input_dim = train_ds[0][0].shape[1] if not n_candles else train_ds[0][0]['main'].shape[1]

    # 🔥 Regression head → output dimension = target length
    target_dim = train_ds[0][1].shape[0]

    model = MultiLineLSTMRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=target_dim,
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
            'output_dim': target_dim,
            'lr': lr,
        }, meta_out)
        print(f"\n✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # --- Evaluation --- #
    if do_validation:
        mse, mae = evaluate_model(model, val_loader)
        if return_val_accuracy:
            return {"mse": mse, "mae": mae}
        
if __name__ == "__main__":
    train_model(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        to_address(pd.read_csv(io.StringIO(
            json_to_csv_in_memory("/home/iatell/projects/meta-learning/data/line_sequence.json")
        ))),
        do_validation=True
    )
