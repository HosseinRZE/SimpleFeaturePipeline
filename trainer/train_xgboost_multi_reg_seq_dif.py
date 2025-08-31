import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import io
import numpy as np
from utils.to_address import to_address
from preprocess.multi_regression_seq_dif import preprocess_sequences_csv_multilines
from utils.json_to_csv import json_to_csv_in_memory
import pandas as pd
import warnings
from add_ons.drop_column import drop_columns
from add_ons.feature_pipeline import FeaturePipeline
def evaluate_model(model, X_val, y_val):
    """
    Evaluate a multi-output regression XGBoost model and print metrics.
    """
    y_pred = model.predict(X_val)

    # Ensure y_val 2D
    if y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    print("\n📊 Validation Report (Multi-Regression):")

    # Compute metrics per target
    mse = mean_squared_error(y_val, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_val, y_pred, multioutput='raw_values')

    # Handle R² carefully
    r2 = []
    for i in range(y_val.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                r2_val = r2_score(y_val[:, i], y_pred[:, i])
            except ValueError:  # e.g., single sample
                r2_val = np.nan
        r2.append(r2_val)
    r2 = np.array(r2)

    # Print per-target metrics
    for i in range(y_val.shape[1]):
        print(f"\nTarget {i}:")
        print(f"  MSE: {mse[i]:.6f}")
        print(f"  MAE: {mae[i]:.6f}")
        print(f"  R² : {r2[i]:.6f}")

    # Global scores
    mse_mean = mean_squared_error(y_val, y_pred)
    mae_mean = mean_absolute_error(y_val, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            r2_mean = r2_score(y_val, y_pred)
        except ValueError:
            r2_mean = np.nan

    print("\n--- Global Scores ---")
    print(f"Mean MSE: {mse_mean:.6f}")
    print(f"Mean MAE: {mae_mean:.6f}")
    print(f"Mean R² : {r2_mean:.6f}")

    return {"mse": mse_mean, "mae": mae_mean, "r2": r2_mean}


def train_model_xgb_multireg(
    data_csv,
    labels_csv,
    model_out_dir="models/saved_models",
    do_validation=True,
    seq_len=1,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    save_model=False,
    return_val_metrics=True,
    **model_params
):
    """
    Train a multi-output XGBoost regressor (one model per output dimension).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/xgb_model_multireg_{timestamp}.pkl"
    meta_out = f"{model_out_dir}/xgb_meta_multireg_{timestamp}.pkl"


    pipeline = FeaturePipeline(
            steps=[
                lambda df: drop_columns(df, ["volume","open","high","close","low"])  # remove volume column
            ],
            norm_methods={
            "main": {"upper_shadow": "standard",}  # normalize directly in df
                }
            )
    # Preprocess dataset for regression
    if do_validation:
        X_train, y_train, X_val, y_val, df, feature_cols = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=True,
            for_xgboost=True,
            debug_sample=True,
            feature_pipeline=pipeline
        )
    else:
        X_train, y_train, df, feature_cols = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=False,
            for_xgboost=True,
            feature_pipeline=pipeline
        )
        X_val, y_val = None, None

    # Base XGBoost regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        **model_params
    )

    # Wrap with MultiOutputRegressor
    model = MultiOutputRegressor(xgb_model, n_jobs=-1)

    # Train
    model.fit(X_train, y_train)

    # Save model & metadata
    if save_model:
        os.makedirs(model_out_dir, exist_ok=True)
        joblib.dump(model, model_out)
        joblib.dump({
            'seq_len': seq_len,
            'feature_cols': feature_cols,
            'target_dim': y_train.shape[1]
        }, meta_out)
        print(f"✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # Evaluate
    val_metrics = None
    if do_validation:
        val_metrics = evaluate_model(model, X_val, y_val)

    if return_val_metrics:
        return val_metrics


if __name__ == "__main__":
    train_model_xgb_multireg(
        "data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        to_address(pd.read_csv(io.StringIO(
        json_to_csv_in_memory("/home/iatell/projects/meta-learning/data/line_sequence.json")
        ))),
        do_validation=True,
        save_model=False
            )
