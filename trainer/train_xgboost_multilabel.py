import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import os
import io
import numpy as np
from preprocess.multilabel_preprocess import preprocess_csv_multilabel
from utils.json_to_csv import json_to_csv_in_memory

def evaluate_multilabel_model(model, X_val, y_val, mlb):
    """
    Evaluate a multi-label XGBoost model and print metrics.
    """
    y_pred = model.predict(X_val)

    print("\n📊 Validation Report (Multi-label):")
    print(classification_report(y_val, y_pred, target_names=mlb.classes_, zero_division=0))

    print("\n🧮 Multi-label Confusion Matrices (per class):")
    mcm = multilabel_confusion_matrix(y_val, y_pred)
    for i, cls in enumerate(mlb.classes_):
        print(f"\nClass '{cls}':")
        print(mcm[i])

    # Exact match ratio (all labels match)
    exact_match = np.all(y_pred == y_val, axis=1).mean()
    print("\nExact match ratio:", exact_match)

    # Micro accuracy (per-label)
    micro_acc = (y_pred == y_val).mean()
    print("Micro accuracy (per-label):", micro_acc)

    return exact_match, micro_acc

def train_model_xgb_multilabel(
    data_csv,
    labels_json,
    model_out_dir="models/saved_models",
    do_validation=True,
    seq_len=1,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    save_model=False,
    return_val_accuracy=True,
    **model_params
):
    """
    Train a multi-label XGBoost model (one-vs-rest using MultiOutputClassifier).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/xgb_model_multilabel_{timestamp}.pkl"
    meta_out = f"{model_out_dir}/xgb_meta_multilabel_{timestamp}.pkl"

    # Convert JSON labels to in-memory CSV
    csv_string = json_to_csv_in_memory(labels_json)
    labels_csv = io.StringIO(csv_string)

    # Preprocess multi-label dataset
    if do_validation:
        X_train, X_val, y_train, y_val, mlb, df, feature_cols = preprocess_csv_multilabel(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=True,
            for_xgboost=True
        )
    else:
        X_train, y_train, mlb, df, feature_cols = preprocess_csv_multilabel(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=False,
            for_xgboost=True
        )
        X_val, y_val = None, None

    # Base XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric='logloss',
        **model_params
    )

    # Wrap with MultiOutputClassifier for multi-label support
    model = MultiOutputClassifier(xgb_model, n_jobs=-1)

    # Train
    model.fit(X_train, y_train)

    # Save model & metadata
    if save_model:
        os.makedirs(model_out_dir, exist_ok=True)
        joblib.dump(model, model_out)
        joblib.dump({
            'seq_len': seq_len,
            'label_classes': mlb.classes_,
            'feature_cols': feature_cols
        }, meta_out)
        print(f"✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # Evaluate
    val_acc_exact, val_acc_micro = None, None
    if do_validation:
        val_acc_exact, val_acc_micro = evaluate_multilabel_model(model, X_val, y_val, mlb)

    if return_val_accuracy:
        return {"exact_match": val_acc_exact, "micro_accuracy": val_acc_micro}

if __name__ == "__main__":
    train_model_xgb_multilabel(
        data_csv="data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        labels_json="data/candle_labels.json",
        do_validation=True
    )
