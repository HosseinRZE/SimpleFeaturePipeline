import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocess.preprocess_final import preprocess_pipeline
from utils.padding_batch_reg import collate_batch
from add_ons.feature_pipeline_base import FeaturePipeline
from utils.generate_name import generate_filenames
from models.neural_nets.vanilla_fnn import VanillaFNN
from models.evaluation.multi_regression import evaluate_model
from models.utils.early_stopping import get_early_stopping_callbacks
from utils.run_debug_mode import run_debug_mode
from utils.save_model_files import save_model_files
from sequencer.sequencer import SequencerAddOn
from add_ons.feature_tracker_addon import FeatureColumnTrackerAddOn
from add_ons.label_padder_add_on import LabelPadder
from add_ons.input_dim_calculator import InputDimCalculator
from add_ons.candle_norm_reduce_addon import CandleNormalizationAddOn
from add_ons.candle_shape_add_on import CandleShapeFeaturesAddOn
from add_ons.drop_column_windowing_add_on import DropColumnsAddOn
from utils.filter_sequences import FilterInvalidSequencesAddOn
from add_ons.prepare_output import PrepareOutputAddOn
from add_ons.RootPower import RootPowerMapperAddOn

# ---------------- Train ---------------- #
def train_model(
    data_csv,
    labels_csv,
    experiment_out_dir="experiments",
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
    scheduler_params={"factor": 0.2, "patience": 5},
    activation_function = "relu",
    use_mse_loss = False,
    use_rescue = False,
    activation_functions = ["relu"]
):
    
    # 2. Create the pipeline and add your modules
    feature_pipeline = FeaturePipeline(
        add_ons=[
            SequencerAddOn(include_cols=None, exclude_cols=None),
            CandleNormalizationAddOn(),
            RootPowerMapperAddOn(
            p=1/4, # Use a root power for aggressive variance increase
            main=["open_prop", "high_prop", "low_prop", "close_prop"], # Apply to features
            y=True                                # Apply to labels
        ),
            DropColumnsAddOn(cols_map={ "main": ["open", "high", "low", "close", "volume"]}),
            FilterInvalidSequencesAddOn(),
            LabelPadder(),
            FeatureColumnTrackerAddOn(), 
            InputDimCalculator(),
            PrepareOutputAddOn()
        ])
    model_save_path, meta_save_path, pipeline_save_path, folder_name = generate_filenames([
        ("model", "pt"),
        ("meta", "pkl"),
        ("pipeline", "pkl"),
        "fnn"
    ])
    feature_pipeline.method_table()
    # Preprocess: pad linePrices and sequences
    if do_validation:
        train_ds, val_ds, returned_state = preprocess_pipeline(
            data_csv, labels_csv,
            val_split=True,
            debug_indices=[0,1,2],
            feature_pipeline=feature_pipeline,
        )
    else:
        train_ds, returned_state = preprocess_pipeline(
            data_csv, labels_csv,
            val_split=True,
            debug_indices=[0,1,2],
            feature_pipeline=feature_pipeline,
        )
        val_ds = None

    input_dim = returned_state['input_dim']
    max_len_y = returned_state['max_len_y']
    feature_columns = returned_state["feature_columns"]

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
    if early_stop:
        # Define the output directory for this specific experiment/model
        model_out_dir = os.path.join(experiment_out_dir, folder_name)
        os.makedirs(model_out_dir, exist_ok=True)
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
        save_model_files(
            base_dir=experiment_out_dir,
            folder_name=folder_name,
            trainer=trainer,
            model_out=model_save_path,
            meta_out=meta_save_path,
            pipeline_out=pipeline_save_path,
            pipeline=feature_pipeline,
            train_model=train_model,
        )

    # --- Evaluation --- #
    if do_validation:
        metrics = evaluate_model(model, val_loader, feature_pipeline)
        if return_val_accuracy:
            return {"accuracy": metrics["mse"] * (-1)}
        
if __name__ == "__main__":
    train_model(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles.csv",
        "/home/iatell/projects/meta-learning/data/baseline_regression.csv",
        do_validation=True,
        test_mode = True,
        max_epochs=100,
        hidden_dim=100,
        lr=0.01,
        batch_size=50,
        optimizer_name= "adamw",
        scheduler_name = "onecycle",
        optimizer_params={},
        scheduler_params={},
        save_model= True,
        use_mse_loss = False,
        use_rescue = False,
       activation_functions=["elu", "elu", "dropout(0.1)", "elu"] #"leaky_relu" "sigmoid" "tanh"  "elu" "relu" "swish" "mish"
    )
