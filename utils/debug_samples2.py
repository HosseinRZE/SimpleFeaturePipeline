from typing import List, Dict, Any
import pandas as pd
from data_structure.sequence_sample import SequenceSample
from data_structure.sequence_collection import SequenceCollection


def _debug_sample_check(indices: List[int], state: Dict[str, Any]):
    """
    Prints detailed information for selected samples within the SequenceCollection.

    This function is meant for debugging the output of the preprocessing pipeline.
    It inspects the specified SequenceSample objects within state['samples'] and 
    prints their metadata, features, and labels in a structured, readable format.

    Args:
        indices (List[int]): List of integer indices (0-based) referring to the
            order of samples inside the SequenceCollection.
        state (Dict[str, Any]): Pipeline state containing a 'samples' key 
            with a SequenceCollection instance.

    Raises:
        ValueError: If 'samples' is missing or is not a SequenceCollection.
    """
    print("\n=== DEBUG SAMPLE CHECK ===")

    samples = state.get("samples", None)
    if samples is None or not isinstance(samples, SequenceCollection):
        raise ValueError(
            "State must contain a 'samples' key holding a SequenceCollection object."
        )

    # --- Access the internal list of samples robustly ---
    try:
        sample_list = samples.get_list()  # if implemented
    except AttributeError:
        # fallback to attribute assumption
        sample_list = getattr(samples, "samples", list(samples))

    total = len(sample_list)
    if total == 0:
        print("⚠️ No samples found in the SequenceCollection.")
        return

    for idx in indices:
        if 0 <= idx < total:
            sample: SequenceSample = sample_list[idx]
            print(f"\n--- Sample {idx} ---")
            print(f"Original index: {sample.original_index}")
            if hasattr(sample, "start_time") and hasattr(sample, "end_time"):
                print(f"Window range: {sample.start_time} → {sample.end_time}")
            else:
                print("Window range: [not available]")
            print(f"y (shape {getattr(sample.y, 'shape', None)}): {sample.y}")

            # Print each feature group
            print("Feature sets:")
            for name, arr in sample.X.items():
                print(f"  • '{name}' →", end=" ")

                if isinstance(arr, pd.DataFrame):
                    print(f"DataFrame {arr.shape}")
                    print("    Columns:", list(arr.columns))
                    print("    Preview:\n", arr.head(5).to_string(index=False))
                elif hasattr(arr, "shape"):
                    print(f"ndarray shape={arr.shape}")
                    print("    Preview:\n", arr[:5])
                elif isinstance(arr, list):
                    print(f"list length={len(arr)}")
                    print("    Preview:", arr[:5])
                else:
                    print(f"type={type(arr).__name__}")
                    print("    Value:", arr)
        else:
            print(f"\n--- Index {idx} is out of bounds (max index = {total-1}) ---")

    print("========================\n")
