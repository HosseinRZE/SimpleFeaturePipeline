import numpy as np
def rocket_transform(X_dicts_list, **kwargs):
    from utils.padd_sequence_xgboost import pad_sequences_dicts
    from sktime.transformations.panel.rocket import Rocket

    strategy = kwargs.get("strategy", "forward_fill")
    num_kernels = kwargs.get("num_kernels", 100)
    normalise = kwargs.get("normalise", False)
    random_state = kwargs.get("random_state", 42)

    # Pad to (n, t, c)
    X_main = pad_sequences_dicts(X_dicts_list, dict_name="main", strategy=strategy)
    X_main_rocket = np.transpose(X_main, (0, 2, 1))

    rocket = Rocket(num_kernels=num_kernels, normalise=normalise, n_jobs=-1, random_state=random_state)
    return rocket.fit_transform(X_main_rocket).values
