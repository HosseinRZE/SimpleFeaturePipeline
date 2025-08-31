import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from preprocess.multi_regression_seq_dif import preprocess_sequences_csv_multilines
from models.LSTM.lstm_multi_line_reg_seq_dif import LSTMMultiRegressor
from utils.print_batch import print_batch
from utils.to_address import to_address
from utils.json_to_csv import json_to_csv_in_memory
from utils.padding_batch_reg import collate_batch
import pandas as pd
import io
import numpy as np

# ---------------- Evaluation ---------------- #
# ---------------- Evaluation ---------------- #
def evaluate_model(model, val_loader, threshold=0.5):
    model.eval()
    all_preds_reg, all_labels_reg = [], []
    all_preds_len, all_labels_len = [], []

    with torch.no_grad():
        for X_batch, y_batch, lengths in val_loader:
            # Send to same device as model
            device = next(model.parameters()).device
            X_batch, y_batch, lengths = (
                X_batch.to(device), y_batch.to(device), lengths.to(device)
            )

            # Forward pass: regression + length logits
            y_pred, len_logits = model(X_batch, lengths)

            # Regression targets
            all_preds_reg.append(y_pred.cpu().numpy())
            all_labels_reg.append(y_batch.cpu().numpy())

            # Length targets
            true_lengths = lengths.cpu().numpy()
            pred_lengths = model.predict_length(len_logits).cpu().numpy()

            all_labels_len.extend(true_lengths.tolist())
            all_preds_len.extend(pred_lengths.tolist())

    # ----- Regression metrics -----
    all_preds_reg = np.vstack(all_preds_reg)
    all_labels_reg = np.vstack(all_labels_reg)

    mse = ((all_preds_reg - all_labels_reg) ** 2).mean()
    mae = np.abs(all_preds_reg - all_labels_reg).mean()

    # ----- Length metrics -----
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(all_labels_len, all_preds_len)
    f1 = f1_score(all_labels_len, all_preds_len, average="macro")

    print("\n📊 Validation Metrics:")
    print(f"  Regression → MSE: {mse:.6f}, MAE: {mae:.6f}")
    print(f"  Length     → Acc: {acc:.4f}, F1: {f1:.4f}")

    return {"mse": mse, "mae": mae, "acc": acc, "f1": f1}


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
    return_val_accuracy = True
):
    from preprocess.multi_regression_seq_dif import preprocess_sequences_csv_multilines
    from torch.utils.data import DataLoader, TensorDataset
    import joblib
    from datetime import datetime
    import os
    import numpy as np

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/lstm_model_multireg_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/lstm_meta_multireg_{timestamp}.pkl"

    # Preprocess: pad linePrices and sequences
    if do_validation:
        train_ds, val_ds, df, feature_cols, max_len_y = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=True,
            for_xgboost=False,
            debug_sample=False
        )
    else:
        train_ds, df, feature_cols, max_len_y = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=False,
            for_xgboost=False,
            debug_sample=False
        )
        val_ds = None

    sample = train_ds[0][0]  # first sample's features
    if isinstance(sample, dict):  # multiple feature groups
        input_dim = sample['main'].shape[1]
    else:  # single tensor
        input_dim = sample.shape[1]

    model = LSTMMultiRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_len_y=max_len_y,
        lr=lr
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_batch) if val_ds else None

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices=1)
    trainer.fit(model, train_loader, val_loader)

    if save_model:
        os.makedirs(model_out_dir, exist_ok=True)
        trainer.save_checkpoint(model_out)
        joblib.dump({
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "max_len_y": max_len_y,
            "feature_cols": feature_cols
        }, meta_out)
        print(f"✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")
    # --- Evaluation --- #
    if do_validation:
        mse, mae, acc, f1 = evaluate_model(model, val_loader)
        if return_val_accuracy:
            return {"mse": mse, "mae": mae, "acc": acc, "f1": f1}
        
if __name__ == "__main__":
    train_model(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        to_address(pd.read_csv(io.StringIO(
            json_to_csv_in_memory("/home/iatell/projects/meta-learning/data/line_sequence.json")
        ))),
        do_validation=True
    )
