import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple
from utils.build_multi_input_dataset import build_multiinput_dataset
from data_structure.sequence_collection import SequenceCollection

def _prepare_output(
    state: Dict[str, Any],
    val_split: bool,
    test_size: float,
    random_state: int,
    for_torch: bool
) -> Tuple:
    """
    Prepares final dataset objects (NumPy arrays or PyTorch datasets) 
    from the pipeline's postprocessed state.

    Supports multi-group feature structures (e.g., {'main', 'aux'}).
    Each group’s sequence lengths are tracked via SequenceSample.x_lengths.
    """

    # --- Retrieve validated structure ---
    samples: SequenceCollection = state.get("samples")
    if not isinstance(samples, SequenceCollection):
        raise TypeError("State must contain a 'samples' key with a SequenceCollection.")

    # --- Extract aligned structures ---
    X_list = [s.X for s in samples]                # list of dicts of DataFrames
    y_list = [s.y for s in samples]                # list of targets
    x_lengths_list = [s.x_lengths for s in samples]  # dict per sample

    # Fallback padded labels (usually set by LabelPadder)
    y_padded = state.get("y_padded", np.array(y_list, dtype=np.float32))

    # --- Handle non-Torch mode ---
    if not for_torch:
        # Flatten only the 'main' feature group (classical ML mode)
        X = np.array([x["main"].to_numpy().flatten() for x in X_list])
        if not val_split:
            return X, y_padded, state

        # --- Split ---
        idx_train, idx_val = train_test_split(
            np.arange(len(X_list)), test_size=test_size, random_state=random_state
        )
        X_train, X_val = X[idx_train], X[idx_val]
        y_train, y_val = y_padded[idx_train], y_padded[idx_val]
        return X_train, y_train, X_val, y_val, state

    # --- Torch mode (multi-input compatible) ---
    if not val_split:
        dataset = build_multiinput_dataset(X_list, y_padded, x_lengths_list)
        return dataset, state

    # --- Split into train/val ---
    indices = np.arange(len(X_list))
    idx_train, idx_val = train_test_split(indices, test_size=test_size, random_state=random_state)

    X_train = [X_list[i] for i in idx_train]
    X_val = [X_list[i] for i in idx_val]
    train_dataset = build_multiinput_dataset(
        X_train, y_padded[idx_train], [x_lengths_list[i] for i in idx_train]
    )
    val_dataset = build_multiinput_dataset(
        X_val, y_padded[idx_val], [x_lengths_list[i] for i in idx_val]
    )

    return train_dataset, val_dataset, state
