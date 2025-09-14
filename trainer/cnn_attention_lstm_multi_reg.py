import sys
from pathlib import Path

# Current notebook location
notebook_path = Path().resolve()

# Add parent folder (meta/) to sys.path
sys.path.append(str(notebook_path.parent))
import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from preprocess.multi_regression_seq_dif3 import preprocess_sequences_csv_multilines
# from models.LSTM.lstm_multi_line_reg_seq_dif import LSTMMultiRegressor
from utils.print_batch import print_batch
from utils.to_address import to_address
from utils.json_to_csv import json_to_csv_in_memory
from utils.padding_batch_reg import collate_batch
import pandas as pd
import io
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score
from add_ons.feature_pipeline5 import FeaturePipeline
from add_ons.drop_column import drop_columns
from add_ons.candle_dif_rate_of_change_percentage2 import add_candle_rocp
from add_ons.candle_proportion import add_candle_proportions
from add_ons.candle_rate_of_change import add_candle_ratios
from utils.make_step import make_step
from models.LSTM.cnn_attention_lstm_multireg import CNNAttentionLSTMMultiRegressor
from scipy.optimize import linear_sum_assignment

# ---------------- Evaluation ---------------- #
def evaluate_model(model, val_loader):
    model.eval()
    all_preds_reg, all_labels_reg = [], []

    with torch.no_grad():
        for X_batch, y_batch, lengths in val_loader:
            device = next(model.parameters()).device
            X_batch = {k: v.to(device) for k, v in X_batch.items()}
            y_batch = y_batch.to(device)
            lengths = lengths.to(device)

            # Forward pass: regression only
            y_pred = model(X_batch, lengths)

            mask = (y_batch != 0).float()

            # --- Hungarian assignment per batch ---
            batch_preds = []
            batch_labels = []
            #y_batch.shape[0] is batch actually
            for i in range(y_batch.shape[0]):
                gt_vals = y_batch[i][mask[i] > 0]  # true targets
                preds = y_pred[i]

                if len(gt_vals) == 0:
                    continue

                cost = torch.cdist(gt_vals.unsqueeze(1), preds.unsqueeze(1), p=2).pow(2)
                row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

                matched_preds = preds[col_ind].cpu().numpy()
                matched_labels = gt_vals[row_ind].cpu().numpy()

                batch_preds.extend(matched_preds.tolist())
                batch_labels.extend(matched_labels.tolist())

            all_preds_reg.extend(batch_preds)
            all_labels_reg.extend(batch_labels)

    # Convert to arrays
    all_preds_reg = np.array(all_preds_reg)
    all_labels_reg = np.array(all_labels_reg)

    # Regression metrics
    mse = ((all_preds_reg - all_labels_reg) ** 2).mean()
    mae = np.abs(all_preds_reg - all_labels_reg).mean()

    print("\n📊 Validation Metrics (Hungarian matched):")
    print(f"  Regression → MSE: {mse:.6f}, MAE: {mae:.6f}")

    return {"mse": mse, "mae": mae}


# ---------------- Train ---------------- #
def train_model(
    data_csv,
    labels_csv,
    model_out_dir="models/saved_models",
    do_validation=True,
    hidden_dim=30,
    num_layers=1,
    lr=0.001,
    batch_size=50,
    max_epochs=100,
    save_model=True,
    return_val_accuracy = True,
    test_mode = False,
    early_stop = False
):

    pipeline = FeaturePipeline(
        steps=[
            # make_step(add_label_normalized_candles),
            make_step(add_candle_rocp),
            make_step(drop_columns, cols_to_drop=["open","high","low","close","volume"]),
            
        ],
        # norm_methods={
        #     "main": {
        #         "upper_shadow": "robust", "body": "standard", "lower_shadow": "standard",
        #         "upper_body_ratio": "standard", "lower_body_ratio": "standard",
        #         "upper_lower_body_ratio": "standard", "Candle_Color": "standard"
        #     }
        # },
        per_window_flags=[
            False, 
          False, 
        #   True
                ]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/lstm_model_multireg_multihead_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/lstm_meta_multireg_multihead_{timestamp}.pkl"

    # Preprocess: pad linePrices and sequences
    if do_validation:
        train_ds, val_ds, df, feature_cols, max_len_y = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=True,
            for_xgboost=False,
            debug_sample=True,
            feature_pipeline=pipeline,
            preserve_order= True
        )
    else:
        train_ds, df, feature_cols, max_len_y = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=False,
            for_xgboost=False,
            debug_sample=False,
            preserve_order= True,
            feature_pipeline=pipeline,
        )
        val_ds = None

    sample = train_ds[0][0]  # first sample's features
    if isinstance(sample, dict):  # multiple feature groups
        input_dim = sample['main'].shape[1]
    else:  # single tensor
        input_dim = sample.shape[1]

    model = CNNAttentionLSTMMultiRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_len_y=max_len_y,
        lr=lr
    )
    init_args = {
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "max_len_y": max_len_y,
    "lr": lr
}

    model_class_info = {
        "module": model.__class__.__module__,
        "class": model.__class__.__name__,
        "init_args": init_args
    }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_batch) if val_ds else None
    # --- Early stopping --- #
    if early_stop == True:
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        early_stop_callback = EarlyStopping(
            monitor="val_loss",   # metric to monitor (must be logged in your LightningModule)
            patience=10,          # number of epochs with no improvement before stopping
            min_delta=0.001,      # minimum improvement to qualify as "better"
            mode="min",           # "min" for loss, "max" for accuracy
            verbose=True
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_out_dir,
            filename="best_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        callbacks=[early_stop_callback,checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        fast_dev_run=test_mode,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks= callbacks if early_stop else None
    )

    trainer.fit(model, train_loader, val_loader)

    # --- Debug / Test mode --- #
    if test_mode:
        save_model = False
        from itertools import islice

        # Try to grab 3rd batch; if not available, take first
        try:
            batch = next(islice(iter(train_loader), 2, 3))
        except StopIteration:
            batch = next(iter(train_loader))

        X_batch_dict, y_batch, lengths = batch

        print("🔍 Debug batch:")
        if isinstance(X_batch_dict, dict):
            print("  Keys in X_batch:", list(X_batch_dict.keys()))
        print("  y_batch shape:", y_batch.shape)
        print("  First label in batch:", y_batch[0])

        # --- Track real column names for each feature group ---
        feature_names_dict = {}
        for name, X_batch in X_batch_dict.items():
            if name == "main":
                # Use actual feature columns after preprocessing
                feature_names_dict[name] = feature_cols
            else:
                # For extra feature groups, fallback to generic names
                feature_names_dict[name] = [f"{name}_{i}" for i in range(X_batch.shape[2])]

        dfs = []
        for name, X_batch in X_batch_dict.items():
            print(f"\nFeature group: {name}")
            print("  X_batch shape:", X_batch.shape)
            print("  First sequence in batch (first  steps):\n", X_batch[0][:])

            batch_size_, seq_len, feature_dim = X_batch.shape
            df_part = pd.DataFrame(
                X_batch.reshape(batch_size_ * seq_len, feature_dim).numpy(),
                columns=feature_names_dict[name]
            )
            dfs.append(df_part)

        # Combine all feature groups horizontally
        global df_seq
        df_seq = pd.concat(dfs, axis=1)
        print("\n✅ Combined df_seq shape:", df_seq.shape)
        print("✅ Column names in df_seq:", df_seq.columns.tolist())

        
    if save_model:
        os.makedirs(model_out_dir, exist_ok=True)
        trainer.save_checkpoint(model_out)
        joblib.dump({
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "max_len_y": max_len_y,
            "feature_cols": feature_cols,
            "scalers": pipeline.scalers,
            "pipeline_config": pipeline.export_config(),
            "model_class_info": model_class_info 
        }, meta_out)
        print(f"✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")


        
    # --- Evaluation --- #
    if do_validation:
        metrics = evaluate_model(model, val_loader)
        if return_val_accuracy:
            return {"accuracy": metrics["mse"] * (-1)}

        
if __name__ == "__main__":
    train_model(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles.csv",
        "/home/iatell/projects/meta-learning/data/line_seq_ordered.csv",
        do_validation=True,
        test_mode = True
    )
