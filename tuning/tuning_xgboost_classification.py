import psutil
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from trainer.train_xgboost_classification import train_model_xgb

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

def train_xgb_tune(config):
    """Single Ray Tune trial."""

    metrics = train_model_xgb(
        "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
        "/home/iatell/projects/meta-learning/data/labeled_ohlcv_string.csv",
        do_validation=True,
        seq_len=config["seq_len"],
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        return_val_accuracy=True,
        save_model=False,
    )

    tune.report(metrics)

def run_tuning(save_model=True):
    """Hyperparameter tuning for XGBoost with Ray Tune."""
    search_space = {
        "seq_len": tune.choice([3, 5, 7]),
        "n_estimators": tune.choice([100, 200, 300]),
        "max_depth": tune.choice([3, 5, 7]),
        "learning_rate": tune.loguniform(1e-3, 0.3),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0)
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        grace_period=1,
        reduction_factor=2
    )

    train_fn_with_resources = tune.with_resources(train_xgb_tune, {"gpu": 1})

    tuner = tune.Tuner(
        train_fn_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=10
        ),
        run_config=tune.RunConfig(
            name="xgb_tuning",
            verbose=1
        ),
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("\n🏆 Best Config:", best_result.config)
    print(f"Best Accuracy: {best_result.metrics['accuracy']:.4f}")

    if save_model:
        print("\n🔁 Retraining best model on full dataset for saving...")
        train_model_xgb(
            "/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles_prop.csv",
            "/home/iatell/projects/meta-learning/data/labeled_ohlcv_string.csv",
            do_validation=False,
            seq_len=best_result.config["seq_len"],
            n_estimators=best_result.config["n_estimators"],
            max_depth=best_result.config["max_depth"],
            learning_rate=best_result.config["learning_rate"],
            subsample=best_result.config["subsample"],
            colsample_bytree=best_result.config["colsample_bytree"],
            save_model=True
        )

if __name__ == "__main__":
    run_tuning(save_model=True)
