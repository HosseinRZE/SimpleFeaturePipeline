import joblib
import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from add_ons.relative_change import add_pct_changes
from add_ons.drop_column import drop_columns
from add_ons.zigzag_single import add_zigzag
from preprocess.classification_pre_dict import preprocess_csv
from models.LSTM.multilstm_classification import MultiLSTMClassifier
from itertools import islice
from add_ons.featue_pipeline2 import FeaturePipeline


# ----------------- Evaluation -----------------
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

# ----------------- Training -----------------
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
    save_model=False,
    return_val_accuracy=False,
    test_mode=True
):
    """
    Train an LSTM classification model with zigzag features and custom normalization.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/lstm_model_class_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/lstm_meta_class_{timestamp}.pkl"

    # --- Define Feature Pipeline ---
    pipeline = FeaturePipeline(
            steps=[lambda df: add_pct_changes(df, separatable="complete")],
            norm_methods={
                "main": {"upper_shadow": "standard"},
                "pct_changes": {"open_pct": "standard", "high_pct": "standard"}
            }
        )

    seq_dict = {"main": 5, "pct_changes": 3}  # different seq lens per group


    # --- Get dataset(s) ---
    if do_validation:
        train_ds, val_ds, label_encoder, df = preprocess_csv(
            data_csv, labels_csv,
            n_candles=seq_dict,
            val_split=True,
            feature_pipeline=pipeline
        )
    else:
        full_dataset,label_encoder, df ,feature_cols= preprocess_csv(
            data_csv, labels_csv,
            n_candles=seq_dict,
            val_split=True,
            feature_pipeline=pipeline
)

    # --- Model config ---
    input_dims = {
        "main": train_ds.X_dict["main"].shape[-1],
        "pct_changes": train_ds.X_dict["pct_changes"].shape[-1]
    }
    num_classes = len(label_encoder.classes_)

    model = MultiLSTMClassifier(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        lr=lr
    )

    # --- DataLoaders ---
    if do_validation:
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=32)
    else:
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = None

    # --- Debug mode ---
    if test_mode:
        X_batch_dict, y_batch = next(islice(iter(train_loader), 2, 3))
        print("🔍 Debug batch (third batch):")
        print("  Keys in X_batch:", list(X_batch_dict.keys()))
        print("  y_batch shape:", y_batch.shape)   # (batch_size,)
        print("  First label in batch:", y_batch[0])

        # Iterate over dict to inspect each input
        for name, X_batch in X_batch_dict.items():
            print(f"\nFeature group: {name}")
            print("  X_batch shape:", X_batch.shape)  # (batch_size, seq_len, feature_dim)
            print("  First sequence in batch:\n", X_batch[0])

            batch_size, seq_len, feature_dim = X_batch.shape
            global df_seq
            df_seq = pd.DataFrame(
                X_batch.reshape(batch_size * seq_len, feature_dim).numpy(),
                columns=[f"{name}_{c}" for c in range(feature_dim)]  # temporary column names
            )


    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        fast_dev_run=test_mode
    )

    trainer.fit(model, train_loader, val_loader)

    # --- Save model & meta ---
    if save_model:
        trainer.save_checkpoint(model_out)
        joblib.dump({
            'input_dim': input_dims,
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

# ----------------- Entry -----------------
if __name__ == "__main__":
    train_model(
        "data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        "data/labeled_ohlcv_string.csv",
        do_validation=True
    )
