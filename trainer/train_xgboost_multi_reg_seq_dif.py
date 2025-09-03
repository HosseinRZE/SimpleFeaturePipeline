import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import io
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

from preprocess.multi_regression_seq_dif import preprocess_sequences_csv_multilines
from add_ons.drop_column import drop_columns
from add_ons.feature_pipeline import FeaturePipeline
from add_ons.dif_seq_candles import add_label_normalized_candles
# ---------------- Evaluation ---------------- #
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import warnings

def evaluate_model(model, length_model, X_val, y_val, true_lengths, return_sequences=False):
    """
    Evaluate multi-output regression with predicted sequence lengths.
    Permutation-invariant: sorts both predictions and true values before computing metrics.
    Can optionally return the predicted vs true sequences for inspection.
    """
    y_pred_full = model.predict(X_val)
    pred_lengths = np.round(length_model.predict(X_val)).astype(int)

    print("\n📊 Validation Report (Multi-Regression with variable-length sequences):")
    mse_list, mae_list, r2_list = [], [], []

    pred_vs_true_list = []  # store predicted vs true sequences if needed

    for i, (pred, pred_len, true_y, true_len) in enumerate(zip(y_pred_full, pred_lengths, y_val, true_lengths)):
        L = min(pred_len, true_len)
        pred_trunc = np.sort(pred[:L])       # sort predictions for permutation-invariant metrics
        true_trunc = np.sort(true_y[:L])     # sort true values

        mse = mean_squared_error(true_trunc, pred_trunc)
        mae = mean_absolute_error(true_trunc, pred_trunc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r2 = r2_score(true_trunc, pred_trunc)
            except ValueError:
                r2 = np.nan

        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)

        print(f"\nSample {i}:")
        print(f"  Predicted length: {pred_len}, True length: {true_len}")
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        print(f"  Predicted lines: {pred_trunc}")
        print(f"  True lines     : {true_trunc}")

        if return_sequences:
            pred_vs_true_list.append((pred_trunc, true_trunc))

    print("\n--- Global Scores ---")
    print(f"Mean MSE: {np.mean(mse_list):.6f}")
    print(f"Mean MAE: {np.mean(mae_list):.6f}")
    print(f"Mean R²: {np.nanmean(r2_list):.6f}")

    results = {"mse": np.mean(mse_list), "mae": np.mean(mae_list), "r2": np.nanmean(r2_list)}
    
    if return_sequences:
        results["pred_vs_true"] = pred_vs_true_list
    
    return results

# ---------------- Train ---------------- #
def train_model_xgb_multireg(
    data_csv,
    labels_csv,
    model_out_dir="models/saved_models",
    do_validation=True,
    n_estimators=1000,
    max_depth=16,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    save_model=False,
    return_val_metrics=True,
    **model_params
):
    """
    Train a multi-output XGBoost regressor with a linked sequence-length predictor.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/xgb_model_multireg_{timestamp}.pkl"
    length_model_out = f"{model_out_dir}/xgb_model_seq_len_{timestamp}.pkl"
    meta_out = f"{model_out_dir}/xgb_meta_multireg_{timestamp}.pkl"

    pipeline = FeaturePipeline(
        steps=[lambda df: add_label_normalized_candles(df, labels_csv),
        lambda df: drop_columns(df, ["volume","open","high","close","low"]),]
               ,
        norm_methods={"main": {"upper_shadow": "standard"}}
    )

    # --- Preprocess data ---
    if do_validation:
        X_train, y_train, X_val, y_val, df, feature_cols, max_len_y, seq_lengths_true = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=True,
            for_xgboost=True,
            debug_sample=True,
            feature_pipeline=pipeline
        )
    else:
        X_train, y_train, df, feature_cols, max_len_y, seq_lengths_true = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=False,
            for_xgboost=True,
            feature_pipeline=pipeline
        )
        X_val, y_val = None, None


    # --- Sequence length targets ---
    if do_validation:
        idx_train, idx_val = train_test_split(
            np.arange(len(seq_lengths_true)),
            test_size=0.2,  # match your preprocess split
            random_state=42
        )
        train_lengths = np.array(seq_lengths_true)[idx_train]
        val_lengths   = np.array(seq_lengths_true)[idx_val]
    else:
        train_lengths = np.array(seq_lengths_true)

    # --- Train max-line regression ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        **model_params
    )
    model = MultiOutputRegressor(xgb_model, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Train length predictor ---
    xgb_len_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        **model_params
    )
    xgb_len_model.fit(X_train, train_lengths)


    # --- Save models ---
    if save_model:
        os.makedirs(model_out_dir, exist_ok=True)
        joblib.dump(model, model_out)
        joblib.dump(xgb_len_model, length_model_out)
        joblib.dump({
            'feature_cols': feature_cols,
            'target_dim': max_len_y
        }, meta_out)
        print(f"✅ Model saved to {model_out}")
        print(f"✅ Length predictor saved to {length_model_out}")
        print(f"✅ Meta saved to {meta_out}")

    # --- Evaluate ---
    val_metrics = None
    if do_validation:
        metrics = evaluate_model(model, xgb_len_model, X_val, y_val, val_lengths, return_sequences=True)


    if return_val_metrics:
        return val_metrics

# ---------------- Main ---------------- #
if __name__ == "__main__":
    train_model_xgb_multireg(
        "data/Bitcoin_BTCUSDT_kaggle_1D_candles.csv",
        "data/seq_line_labels.csv",
        do_validation=True,
        save_model=False
    )
