import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score
import os
import io
import numpy as np
from preprocess.multilabel_preprocess import preprocess_csv_multilabel
from utils.json_to_csv import json_to_csv_in_memory
from utils.multilabel_threshold_tuning import tune_thresholds
def evaluate_multilabel_model(model, X_val, y_val, mlb, thresholds=None):
    """
    Evaluate a multi-label XGBoost model and print metrics.
    Optionally apply per-label thresholds.
    """
    # Predict probabilities per label
    y_probs = np.column_stack([est.predict_proba(X_val)[:, 1] for est in model.estimators_])

    # Apply thresholds
    if thresholds is None:
        thresholds = [0.5] * y_val.shape[1]
    y_pred = np.zeros_like(y_val)
    for i, t in enumerate(thresholds):
        y_pred[:, i] = (y_probs[:, i] >= t).astype(int)

    print("\n📊 Validation Report (Multi-label):")
    print(classification_report(y_val, y_pred, target_names=mlb.classes_, zero_division=0))

    print("\n🧮 Multi-label Confusion Matrices (per class):")
    mcm = multilabel_confusion_matrix(y_val, y_pred)
    for i, cls in enumerate(mlb.classes_):
        print(f"\nClass '{cls}':")
        print(mcm[i])

    exact_match = np.all(y_pred == y_val, axis=1).mean()
    print("\nExact match ratio:", exact_match)

    micro_acc = (y_pred == y_val).mean()
    print("Micro accuracy (per-label):", micro_acc)

    return exact_match, micro_acc, y_probs


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
    label_weighting="none",  # "none", dict, or "scale_pos"
    threshold_tuning = True,
    **model_params
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/xgb_model_multilabel_{timestamp}.pkl"
    meta_out = f"{model_out_dir}/xgb_meta_multilabel_{timestamp}.pkl"

    csv_string = json_to_csv_in_memory(labels_json)
    labels_csv = io.StringIO(csv_string)

    if do_validation:
        X_train, X_val, y_train, y_val, mlb, df, feature_cols, label_weights = preprocess_csv_multilabel(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=True,
            for_xgboost=True,
            debug_sample=[10, 15],
            label_weighting=label_weighting
        )
    else:
        X_train, y_train, mlb, df, feature_cols, label_weights = preprocess_csv_multilabel(
            data_csv, labels_csv,
            n_candles=seq_len,
            val_split=False,
            for_xgboost=True,
            label_weighting=label_weighting
        )
        X_val, y_val = None, None

    xgb_models = []
    for w in label_weights:
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric='logloss',
            scale_pos_weight=w,
            **model_params
        )
        xgb_models.append(xgb_model)

    model = MultiOutputClassifier(xgb_models[0], n_jobs=-1)
    model.estimators_ = xgb_models
    model.fit(X_train, y_train)

    # Tune thresholds if validation set exists
    optimal_thresholds = None
    val_acc_exact, val_acc_micro = None, None
    if do_validation:
        # --- Step 1: Predict probabilities once ---
        y_probs = np.column_stack([est.predict_proba(X_val)[:, 1] for est in model.estimators_])

        # --- Step 2: Evaluate with default threshold 0.5 ---
        val_acc_exact_default, val_acc_micro_default, _ = evaluate_multilabel_model(
            model, X_val, y_val, mlb, thresholds=[0.5]*y_val.shape[1]
        )
        if threshold_tuning:
        # --- Step 3: Tune optimal thresholds per label ---
            optimal_thresholds = tune_thresholds(y_val, y_probs)
            print("\n📌 Optimal thresholds per label:", dict(zip(mlb.classes_, optimal_thresholds)))

            # --- Step 4: Evaluate with tuned thresholds ---
            val_acc_exact_tuned, val_acc_micro_tuned, _ = evaluate_multilabel_model(
                model, X_val, y_val, mlb, thresholds=optimal_thresholds
            )

    if save_model:
        os.makedirs(model_out_dir, exist_ok=True)
        joblib.dump(model, model_out)
        joblib.dump({
            'seq_len': seq_len,
            'label_classes': mlb.classes_,
            'feature_cols': feature_cols,
            'optimal_thresholds': optimal_thresholds
        }, meta_out)
        print(f"✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    if return_val_accuracy:
        return {
            "exact_match": val_acc_exact,
            "micro_accuracy": val_acc_micro,
            "label_weights": label_weights,
            "optimal_thresholds": optimal_thresholds
        }


if __name__ == "__main__":
    train_model_xgb_multilabel(
        data_csv="data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        labels_json="data/candle_labels.json",
        do_validation=True,
        label_weighting="scale_pos"
    )
