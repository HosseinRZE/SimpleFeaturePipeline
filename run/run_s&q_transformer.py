import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocess.preprocess_final import preprocess_pipeline
from utils.padding_batch_reg import collate_batch
from feature_pipeline.feature_pipeline_base import FeaturePipeline
from utils.generate_name import generate_filenames
from models.neural_nets.transformer import TransformerWithPositionalEncoding
from models.evaluation.multi_regression import evaluate_model
from models.utils.early_stopping import get_early_stopping_callbacks
from utils.run_debug_mode import run_debug_mode
from utils.save_model_files import save_model_files
from sequencer.sequencer import SequencerAddOn
from add_ons.feature_tracker_addon import FeatureColumnTrackerAddOn
from add_ons.candle_proportion_simple import CandleShapeFeaturesAddOn
from add_ons.input_dim_calculator import InputDimCalculator
from add_ons.candle_normalization_addon import CandleNormalizationAddOn
from add_ons.drop_column_windowing_add_on import DropColumnsAddOn
from add_ons.real_price_multiplier import RealPriceMultiplier
from add_ons.value_extender import ValueExtenderAddOn
from utils.filter_sequences import FilterInvalidSequencesAddOn
from add_ons.prepare_output import PrepareOutputAddOn
from add_ons.RootPower import RootPowerMapperAddOn
from add_ons.ArcTanMapper import ArctanMapperAddOn
from add_ons.pct_change import PctChangeMapperAddOn
from add_ons.price_rate_change import PriceRateChange
from add_ons.universal_scaler_add_on import ScalerMapperAddOn

def train_model(
    data_csv,
    labels_csv,
    experiment_out_dir="experiments",
    do_validation=True,
    hidden_dim=10,
    lr=0.001,
    batch_size=50,
    max_epochs=200,
    save_model=True,
    return_val_accuracy = True,
    test_mode = False,
    early_stop = False,
    num_layers=1,
    positional_encoding="sinusoidal", 
    num_heads=4,
    feedforward_dim=128,
    attention_name = "tanh_attention",
    optimizer_name= "adamw",
    first_drop = 0.3,
    scheduler_name = "reduce_on_plateau",
    optimizer_params={"weight_decay": 0.01},
    scheduler_params={"factor": 0.2, "patience": 5},
    use_mse_loss = False,
    bidirectional = False
):
    # 2. Create the pipeline and add your modules
    feature_pipeline = FeaturePipeline(
        add_ons=[
            SequencerAddOn(include_cols=None, exclude_cols=None),
            CandleShapeFeaturesAddOn(seperatable="complete"),
            CandleNormalizationAddOn(),
            PriceRateChange(),
            # ScalerMapperAddOn(
            #     method="standard",
            #     y=True,
            #     features={"main": ["open_prop", "high_prop", "low_prop", "close_prop"]},),
            ArctanMapperAddOn(
                a=3,
                b=1,  # Use a root power for aggressive variance increase
            target_features={"main":["open_prop", "high_prop", "low_prop", "close_prop"]}, # Apply to features
            y=True,
        ),
            # RootPowerMapperAddOn(
            #     p=0.33,
            #     b=1,
            #     target_features={"main": ["open_prop", "high_prop", "low_prop", "close_prop"]},
            #     transform_y=True,
            # ),
            DropColumnsAddOn(cols_map={ "main": ["open", "high", "low", "close", "volume"]}),
            FilterInvalidSequencesAddOn(),
            FeatureColumnTrackerAddOn(), 
            InputDimCalculator(),
            # ValueExtenderAddOn(n=4, v=0.0),
            PrepareOutputAddOn(metadata_keys=["last_close_price"]),
            RealPriceMultiplier()
        ])
    model_save_path, meta_save_path, pipeline_save_path, folder_name = generate_filenames([
        ("model", "ckpt"),
        ("meta", "pkl"),
        ("pipeline", "pkl"),
        "CnnLstmAttention"
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
    print("returned_state[max_len_y]",returned_state["max_len_y"])
    input_dim = returned_state['input_dim']
    # max_len_y = 5
    max_len_y = returned_state["max_len_y"]
    feature_columns = returned_state["feature_columns"]
    model = TransformerWithPositionalEncoding(
        input_dim=input_dim,
        hidden_dim = hidden_dim,
        num_layers=num_layers,
        max_len_y=max_len_y,
        lr=lr,
        positional_encoding = positional_encoding, 
        num_heads = num_heads,
        feedforward_dim = feedforward_dim,
        optimizer_name= optimizer_name,
        first_drop = first_drop,
        scheduler_name = scheduler_name,
        optimizer_params= optimizer_params,
        scheduler_params= scheduler_params 
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
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

    # --- Evaluation --- #
    if do_validation:
        metrics = evaluate_model(model, val_loader, feature_pipeline)

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

    if return_val_accuracy:
        return {"accuracy": metrics["mse"] * (-1)}
    
if __name__ == "__main__":
    train_model(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles.csv",
        "/home/iatell/projects/meta-learning/data/line_seq_ordered_added.csv",
        do_validation=True,
        test_mode = False,
        max_epochs=200,
        hidden_dim=100,
        lr=0.001,
        batch_size=50,
        optimizer_name= "adamw",
        scheduler_name = None,
        optimizer_params={},
        scheduler_params={},
        save_model= True,
        use_mse_loss = False,

    )
