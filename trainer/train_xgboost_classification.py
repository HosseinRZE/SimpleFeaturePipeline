import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from preprocess.classification_pre import preprocess_csv

def train_model_xgb(
    data_csv,
    labels_csv,
    model_out_dir="models/saved_models",
    do_validation=False,
    seq_len=3,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    save_model=True,
    return_val_accuracy=False,
    **model_params
):
    """
    Train an XGBoost classification model on GPU.

    Args:
        data_csv (str): Path to candles CSV.
        labels_csv (str): Path to labels CSV.
        model_out_dir (str): Folder to save model & metadata.
        do_validation (bool): Whether to split data for validation.
        seq_len (int): Number of candles per sequence.
        n_estimators (int): Number of boosting rounds.
        max_depth (int): Maximum depth of each tree.
        learning_rate (float): Learning rate (eta).
        subsample (float): Subsample ratio of training instances.
        colsample_bytree (float): Subsample ratio of columns per tree.
        save_model (bool): Whether to save model.
        return_val_accuracy (bool): Whether to return validation accuracy.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/xgb_model_class_{timestamp}.pkl"
    meta_out  = f"{model_out_dir}/xgb_meta_class_{timestamp}.pkl"
    pipeline = FeaturePipeline(
            steps=[lambda df: drop_columns(df, ["open","high","close","volume", "low",
])],
    # --- Dataset ---
    if do_validation:
        X_train, y_train, X_val, y_val, label_encoder, df = preprocess_csv(
            data_csv, labels_csv, n_candles=seq_len, val_split=True, for_xgboost=True
        )
    else:
        X_train, y_train, label_encoder, df = preprocess_csv(
            data_csv, labels_csv, n_candles=seq_len, val_split=False, for_xgboost=True
        )
        X_val, y_val = None, None

    # --- Model ---
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        device='cuda',       # Use GPU
        eval_metric='mlogloss',
        use_label_encoder=False,
        **model_params
    )

    # --- Training ---
    if do_validation:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    # --- Save Model & Meta ---
    if save_model:
        joblib.dump(model, model_out)
        joblib.dump({
            'seq_len': seq_len,
            'label_classes': label_encoder.classes_
        }, meta_out)
        print(f"✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # --- Validation Evaluation ---
    val_acc = None
    if do_validation:
        y_pred = model.predict(X_val)
        val_acc = (y_pred == y_val).mean()

        print("\n📊 Validation Report:")
        print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

    if return_val_accuracy:
        return {"accuracy": val_acc}


if __name__ == "__main__":
    train_model_xgb(
        "data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        "data/labeled_ohlcv_candle.csv",
        do_validation=True
    )
