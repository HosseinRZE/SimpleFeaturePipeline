import numpy as np

def print_scaler_dict(name, scalers_dict):
    print(f"\n🔹 {name}:")
    for k, s in scalers_dict.items():
        if hasattr(s, "mean_") and hasattr(s, "var_"):
            print(f"  {k}: mean={np.round(s.mean_, 4)}, var={np.round(s.var_, 4)}")
        else:
            print(f"  {k}: (no mean_/var_ found) -> {s}")