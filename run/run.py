import dill  
import joblib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from preprocess.multi_regression_seq_dif4 import preprocess_sequences_csv_multilines
# from models.LSTM.lstm_multi_line_reg_seq_dif import LSTMMultiRegressor
from utils.print_batch import print_batch
from utils.to_address import to_address
from utils.json_to_csv import json_to_csv_in_memory
from utils.padding_batch_reg import collate_batch
import pandas as pd
import io
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score
from add_ons.feature_pipeline_base import FeaturePipeline
from add_ons.drop_columns2 import drop_columns
from add_ons.candle_dif_rate_of_change_percentage2 import add_candle_rocp
from add_ons.candle_proportion import add_candle_proportions
from add_ons.candle_rate_of_change import add_candle_ratios
from add_ons.candle_proportion_simple import add_candle_shape_features
from add_ons.normalize_candle_seq import add_label_normalized_candles
from utils.make_step import make_step
from utils.print_scalers import print_scaler_dict
from scipy.optimize import linear_sum_assignment
from add_ons.after_burner.universal_scaler import universal_scaler
from models.neural_nets.vanilla_fnn import VanillaFNN
from models.evaluation.multi_regression import evaluate_model
from models.utils.early_stopping import get_early_stopping_callbacks
from utils.run_debug_mode import run_debug_mode
# ---------------- Train ---------------- #
def train_model(
    data_csv,
    labels_csv,
    model_out_dir="models/saved_models",
    do_validation=True,
    hidden_dim=10,
    lr=0.0001,
    batch_size=50,
    max_epochs=200,
    save_model=True,
    return_val_accuracy = True,
    test_mode = False,
    early_stop = False,
    optimizer_name= "adamw",
    scheduler_name = "reduce_on_plateau",
    optimizer_params={"weight_decay": 0.01},
    scheduler_params={"factor": 0.2, "patience": 5} ,
    activation_function = "relu",
    scale_labels = False,
    use_mse_loss = False,
    use_rescue = False,
    activation_functions = ["relu"]
):

    # 2. Create the pipeline and add your modules
    pipeline = FeaturePipeline(
        # add_ons=[scaler_addon, xgb_addon],
        extra_params={'some_custom_param': 123}
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out = f"{model_out_dir}/fnn_multihead_{timestamp}.pt"
    meta_out  = f"{model_out_dir}/fnn_multireg_multihead_{timestamp}.pkl"
    pipeline_out = f"{model_out_dir}/feature_pipeline_{timestamp}.pkl"
    # Preprocess: pad linePrices and sequences
    if do_validation:
        train_ds, val_ds, df, feature_columns, max_len_y = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=True,
            for_xgboost=False,
            debug_sample=[0,1,2],
            feature_pipeline=pipeline,
            preserve_order= True,
            scale_labels = scale_labels
        )
    else:
        train_ds, df, feature_columns, max_len_y = preprocess_sequences_csv_multilines(
            data_csv, labels_csv,
            val_split=False,
            for_xgboost=False,
            debug_sample=False,
            preserve_order= True,
            feature_pipeline=pipeline,
            scale_labels = scale_labels
        )
        val_ds = None

    sample = train_ds[0][0]  # first sample's features
    if isinstance(sample, dict):
        # build a dict of input_dims for all feature groups
        input_dim = {k: v.shape[1] for k, v in sample.items()}
    else:
        # single tensor → wrap into dict with a default key
        input_dim = {"main": sample.shape[1]}

    model = VanillaFNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=max_len_y,
        lr=lr,
        optimizer_name= optimizer_name,
        scheduler_name = scheduler_name,
        optimizer_params= optimizer_params,
        scheduler_params= scheduler_params ,
        use_mse_loss = use_mse_loss,
        activation_functions = activation_functions,
        use_rescue = use_rescue
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_batch) if val_ds else None
    
    callbacks = get_early_stopping_callbacks(model_out_dir) if early_stop else []

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        log_every_n_steps= 3,
        devices=1,
        fast_dev_run=test_mode,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks= callbacks if early_stop else None
    )

    trainer.fit(model, train_loader, val_loader)

    # --- Debug / Test mode --- #
    if test_mode:
        save_model, df_seq = run_debug_mode(train_loader, feature_columns, test_mode)

    if save_model:
        os.makedirs(model_out_dir, exist_ok=True)
        trainer.save_checkpoint(model_out)
        joblib.dump({
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "max_len_y": max_len_y,
            "feature_columns": feature_columns,
            "scalers": pipeline.scalers,
            "window_scalers": pipeline.window_scalers,
            "pipeline_config": pipeline.export_config(),
            "target_scalers": pipeline.export_target_scalers(),
        }, meta_out)
        print(f"✅ Model saved to {model_out}")
        print(f"✅ Meta saved to {meta_out}")
        # 3. Save entire FeaturePipeline object
        try:
            joblib.dump(pipeline, pipeline_out)
            print(f"✅ Full FeaturePipeline saved to {pipeline_out}")
        except Exception as e:
            # Fallback to dill if joblib can't serialize some parts
            pipeline_out = pipeline_out.replace(".pkl", ".dill")
            with open(pipeline_out, "wb") as f:
                dill.dump(pipeline, f)
            print(f"⚠️ joblib failed, saved pipeline with dill instead → {pipeline_out}")
        
    # --- Evaluation --- #
    if do_validation:
        metrics = evaluate_model(model, val_loader,pipeline)
        if return_val_accuracy:
            return {"accuracy": metrics["mse"] * (-1)}

        
if __name__ == "__main__":
    train_model(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles.csv",
        "/home/iatell/projects/meta-learning/data/baseline_regression.csv",
        do_validation=True,
        test_mode = False,
        scale_labels = False,
        max_epochs=200,
        hidden_dim=100,
        lr=0.01,
        batch_size=50,
        optimizer_name= "adamw",
        scheduler_name = "onecycle",
        optimizer_params={},
        scheduler_params={},
        save_model= False,
        use_mse_loss = False,
        use_rescue = False,
       activation_functions=["elu", "elu", "dropout(0.1)", "elu"] #"leaky_relu" "sigmoid" "tanh"  "elu" "relu" "swish" "mish"
    )
