import numpy as np

def debug_print_samples(indices, X_dicts_list=None, X_trans=None, y_arr=None, mlb=None):
    """Print input(s) and corresponding labels for debugging."""
    for idx in indices:
        print(f"\n--- DEBUG SAMPLE {idx} ---")
        print("Target (raw vector):", y_arr[idx])
        if mlb is not None:
            active = [mlb.classes_[i] for i in np.where(y_arr[idx] == 1)[0]]
            print("Active labels:", active)

        if X_trans is not None:
            # XGBoost mode: transformed feature vector
            print("Transformed features shape:", X_trans[idx].shape)
            print("First few features:", X_trans[idx][:10])
        elif X_dicts_list is not None:
            # Torch mode: print dicts of subsequences
            for dict_name, arr in X_dicts_list[idx].items():
                print(f"[{dict_name}] Shape:", arr.shape)
                print(f"[{dict_name}] First few rows:\n", arr[:5])
        print("------------------------------")
