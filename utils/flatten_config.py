def flatten_config(config):
    """
    Flattens nested Ray Tune configs into kwargs for train_model.
    Example:
        {"optimizer_params": {"weight_decay": 0.01}}
    →  {"weight_decay": 0.01}
    """
    flat = {}
    for k, v in config.items():
        if isinstance(v, dict):
            flat.update(v)   # flatten subdict
        else:
            flat[k] = v
    return flat
