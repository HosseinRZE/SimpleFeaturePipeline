import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from preprocess.classification_pre import preprocess_csv

SEQ_LEN = 3
def train_model_xgb(
    data_csv,
    labels_csv,
    model_out_dir="models/saved_models",
    do_validation=False
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/xgb_model_class_{timestamp}.pkl"
    meta_out  = f"{model_out_dir}/xgb_meta_class_{timestamp}.pkl"

    if do_validation:
        X_train, y_train, X_val, y_val, label_encoder, df = preprocess_csv(
            data_csv, labels_csv, n_candles=SEQ_LEN, val_split=True, for_xgboost=True
        )
    else:
        X_train, y_train, label_encoder, df = preprocess_csv(
            data_csv, labels_csv, n_candles=SEQ_LEN, val_split=False, for_xgboost=True
        )
        X_val, y_val = None, None

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False
    )

    if do_validation:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
    else:
        model.fit(X_train, y_train)

    joblib.dump(model, model_out)
    joblib.dump({
        'seq_len': SEQ_LEN,
        'label_classes': label_encoder.classes_
    }, meta_out)

    print(f"✅ Model saved to {model_out}")
    print(f"✅ Meta saved to {meta_out}")

    if do_validation:
        y_pred = model.predict(X_val)
        print("\n📊 Validation Report:")
        print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

if __name__ == "__main__":
    train_model_xgb(
        "data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        "data/labeled_ohlcv_string.csv",
        do_validation=True
    )