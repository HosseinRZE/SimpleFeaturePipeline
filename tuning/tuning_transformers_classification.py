import os
import torch
import psutil
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from trainer.train_transformer_classification import train_model

def resource_usage():
    """Print current CPU, RAM, and GPU usage."""
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    usage = f"💻 CPU: {cpu:.1f}% | 🧠 RAM: {ram:.1f}%"
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            usage += f" | 🎮 GPU: {gpus[0].load*100:.1f}% VRAM: {gpus[0].memoryUtil*100:.1f}%"
    except ImportError:
        pass
    print(usage)

def train_transformer_tune(config):
    """
    Single Ray Tune trial for TransformerClassifier.
    Args:
        config (dict): hyperparameters for this trial.
    """
    resource_usage()

    metrics = train_model(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        "/home/iatell/projects/meta-learning/data/labeled_ohlcv_string.csv",
        do_validation=True,
        model_out_dir="models/saved_models",
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        save_model=False,
        return_val_accuracy=True
    )

    tune.report(metrics)

def run_tuning(save_model=True):
    """Hyperparameter tuning for TransformerClassifier with Ray Tune."""

    search_space = {
        "seq_len": tune.choice([3, 5, 7]),
        "d_model": tune.choice([32, 64, 128]),
        "num_heads": tune.choice([1, 2, 4]),
        "num_layers": tune.choice([1, 2, 3]),
        "dim_feedforward": tune.choice([64, 128, 256]),
        "dropout": tune.choice([0.1, 0.2, 0.3]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "max_epochs": tune.choice([5, 10, 15])
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        train_transformer_tune,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=10
        ),
        run_config=tune.RunConfig(
            name="transformer_tuning",
            verbose=1
        )
    )

    results = tuner.fit()

    # Best trial
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("\n🏆 Best Config:", best_result.config)
    print(f"Best Accuracy: {best_result.metrics['accuracy']:.4f}")

    if save_model:
        print("\n🔁 Retraining best model on full dataset for saving...")
        train_model(
            "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
            "/home/iatell/projects/meta-learning/data/labeled_ohlcv_string.csv",
            do_validation=False,
            model_out_dir="models/saved_models",
            seq_len=best_result.config["seq_len"],
            d_model=best_result.config["d_model"],
            num_heads=best_result.config["num_heads"],
            num_layers=best_result.config["num_layers"],
            dim_feedforward=best_result.config["dim_feedforward"],
            dropout=best_result.config["dropout"],
            lr=best_result.config["lr"],
            batch_size=best_result.config["batch_size"],
            max_epochs=best_result.config["max_epochs"],
            save_model=True
        )

if __name__ == "__main__":
    run_tuning(save_model=True)
