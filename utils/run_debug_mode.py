import pandas as pd
from itertools import islice

def run_debug_mode(train_loader, feature_columns, test_mode=False):
    """
    If test_mode is True, inspects a small batch from the training loader
    and prints debug information about feature groups and batch shapes.

    Args:
        train_loader (DataLoader): Your training data loader.
        feature_columns (dict): Mapping from feature group name -> column names.
        test_mode (bool): If True, runs in debug/test mode.

    Returns:
        tuple: (save_model, df_seq) where:
            - save_model (bool): Flag indicating whether to save the model.
            - df_seq (pd.DataFrame | None): Combined dataframe of feature groups (if test_mode=True).
    """
    if not test_mode:
        return True, None  # normal mode, saving is allowed

    save_model = False  # disable saving during test mode

    # Try to grab 3rd batch; if not available, take first
    try:
        batch = next(islice(iter(train_loader), 2, 3))
    except StopIteration:
        batch = next(iter(train_loader))

    X_batch_dict, y_batch, lengths, _ = batch

    print("🔍 Debug batch:")
    if isinstance(X_batch_dict, dict):
        print("  Keys in X_batch:", list(X_batch_dict.keys()))
    print("  y_batch shape:", y_batch.shape)
    print("  First label in batch:", y_batch[0])

    # --- Track real column names for each feature group ---
    feature_names_dict = {}
    for name, X_batch in X_batch_dict.items():
        if name in feature_columns:
            feature_names_dict[name] = feature_columns[name]
        else:
            feature_names_dict[name] = [f"{name}_{i}" for i in range(X_batch.shape[2])]

    dfs = []
    for name, X_batch in X_batch_dict.items():
        print(f"\nFeature group: {name}")
        print("  X_batch shape:", X_batch.shape)
        print("  First sequence in batch (first steps):\n", X_batch[0][:])

        batch_size_, seq_len, feature_dim = X_batch.shape
        df_part = pd.DataFrame(
            X_batch.reshape(batch_size_ * seq_len, feature_dim).numpy(),
            columns=feature_names_dict[name]
        )
        dfs.append(df_part)

    # Combine all feature groups horizontally
    df_seq = pd.concat(dfs, axis=1)
    print("\n✅ Combined df_seq shape:", df_seq.shape)
    print("✅ Column names in df_seq:", df_seq.columns.tolist())

    return save_model, df_seq
