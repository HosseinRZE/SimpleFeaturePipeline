import os
import torch
import psutil
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from trainer.train_cnn_lstm_classification import train_model


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


def train_cnn_lstm_tune(config):
    """
    Single Ray Tune trial.
    Args:
        config (dict): hyperparameters for this trial.
    """
    resource_usage()  # Show current hardware usage

    # Train using existing train_model function
    metrics = train_model(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        "/home/iatell/projects/meta-learning/data/labeled_ohlcv_string.csv",
        do_validation=True,
        model_out_dir="models/saved_models",
        seq_len=config["seq_len"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        cnn_channels=config["cnn_channels"],
        cnn_kernel_sizes=config["cnn_kernel_sizes"],
        cnn_strides=config["cnn_strides"],
        cnn_paddings=config["cnn_paddings"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        return_val_accuracy=True,  # Expects dict with "accuracy" and optionally "loss"
        save_model=False  # Never save during search
    )

    # Report metrics to Ray Tune
    tune.report(**metrics)


def run_tuning(save_model=True):
    """Hyperparameter tuning for CNN LSTM with Ray Tune."""

    search_space = {
        "seq_len": tune.choice([3, 5, 7]),
        "hidden_dim": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "max_epochs": tune.choice([5, 10, 15]),
        # CNN knobs – keep all tuples the same length
        "cnn_channels": tune.choice([(32,), (64,), (32, 64), (64, 32)]),
        "cnn_kernel_sizes": tune.choice([(2,), (3,), (2, 2), (3, 3)]),
        "cnn_strides": tune.choice([(1,), (1, 1)]),
        "cnn_paddings": tune.choice([(0,), (0, 0)]),
    }

    scheduler = ASHAScheduler(
        metric="accuracy",  # must exist in metrics dict from train_model
        mode="max",
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        train_cnn_lstm_tune,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=10
        ),
        run_config=tune.RunConfig(
            name="cnn_lstm_tuning",
            verbose=1
        )
    )

    results = tuner.fit()

    # Best trial
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("\n🏆 Best Config:", best_result.config)
    print(f"Best Accuracy: {best_result.metrics['accuracy']:.4f}")

    # Optional: retrain best model on full data and save
    if save_model:
        print("\n🔁 Retraining best model on full dataset for saving...")
        train_model(
            "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
            "/home/iatell/projects/meta-learning/data/labeled_ohlcv_string.csv",
            do_validation=False,
            model_out_dir="models/saved_models",
            seq_len=best_result.config["seq_len"],
            hidden_dim=best_result.config["hidden_dim"],
            num_layers=best_result.config["num_layers"],
            cnn_channels=best_result.config["cnn_channels"],
            cnn_kernel_sizes=best_result.config["cnn_kernel_sizes"],
            cnn_strides=best_result.config["cnn_strides"],
            cnn_paddings=best_result.config["cnn_paddings"],
            lr=best_result.config["lr"],
            batch_size=best_result.config["batch_size"],
            max_epochs=best_result.config["max_epochs"],
            save_model=True
        )


if __name__ == "__main__":
    run_tuning(save_model=True)
