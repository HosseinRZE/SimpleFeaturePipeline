from itertools import islice
import pandas as pd
def print_batch(train_loader, feature_cols, batch_idx=2):
    """
    Pretty-print a batch of sequences (for debugging).

    Args:
        train_loader: DataLoader to draw a batch from.
        feature_cols: List of column names (single-DF case) or dict of column names (multi-DF).
        batch_idx (int, optional): Which batch index to fetch (default=2 = third batch).
    Returns:
        DataFrame (df_seq) or None if batch not available.
    """
    try:
        batch = next(islice(iter(train_loader), batch_idx, batch_idx + 1))
    except StopIteration:
        print(f"⚠️ No batch at index {batch_idx} (DataLoader too small).")
        return None

    if isinstance(batch[0], dict):  # multi-dict case
        X_batch_dict, y_batch = batch
        print(f"\n🔍 Debug batch #{batch_idx}")
        print("  Keys in X_batch:", list(X_batch_dict.keys()))
        print("  y_batch shape:", y_batch.shape)
        print("  First label in batch:", y_batch[0].item())

        dfs = []
        for name, X_batch in X_batch_dict.items():
            print(f"\nFeature group: {name}")
            print("  X_batch shape:", X_batch.shape)
            print("  First sequence in batch:\n", X_batch[0])

            batch_size, seq_len, feature_dim = X_batch.shape
            df_part = pd.DataFrame(
                X_batch.reshape(batch_size * seq_len, feature_dim).numpy(),
                columns=feature_cols[name]  # ✅ dict of names
            )
            dfs.append(df_part)

        df_seq = pd.concat(dfs, axis=1)
        print("\n✅ Combined df_seq shape:", df_seq.shape)
        return df_seq

    else:  # single-DF case
        X_batch, y_batch = batch
        print(f"\n🔍 Debug batch #{batch_idx}")
        print("  X_batch shape:", X_batch.shape)
        print("  y_batch shape:", y_batch.shape)
        print("  First label in batch:", y_batch[0].item())
        print("  First sequence in batch:\n", X_batch[0])

        batch_size, seq_len, feature_dim = X_batch.shape
        df_seq = pd.DataFrame(
            X_batch.reshape(batch_size * seq_len, feature_dim).numpy(),
            columns=feature_cols  # ✅ plain list of names
        )
        print("\n✅ df_seq shape:", df_seq.shape)
        return df_seq
