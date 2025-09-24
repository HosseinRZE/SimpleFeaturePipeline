import numpy as np


def flatten_transform(X_dicts_list, **kwargs):
    from utils.padd_sequence_xgboost import pad_sequences_dicts

    strategy = kwargs.get("strategy", "forward_fill")
    X_main = pad_sequences_dicts(X_dicts_list, dict_name="main", strategy=strategy)
    n, t, c = X_main.shape
    return X_main.reshape(n, t * c)