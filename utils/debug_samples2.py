from typing import List, Dict, Any, Tuple

def _debug_sample_check(indices: List[int], state: Dict[str, Any]):
    """Prints specified samples for debugging."""
    print("\n=== DEBUG SAMPLE CHECK ===")
    X_list = state.get('X_list', [])
    y_list = state.get('y_list', [])

    for idx in indices:
        if 0 <= idx < len(X_list):
            print(f"\n--- Sample {idx} ---")
            print(f"Label: {y_list[idx]}")
            for name, arr in X_list[idx].items():
                print(f"  Features '{name}' (shape: {arr.shape}):\n{arr[:]}") # Print first 5 rows
        else:
            print(f"\n--- Index {idx} is out of bounds ---")
    print("========================\n")