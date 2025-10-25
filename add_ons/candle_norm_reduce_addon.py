from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_sample import SequenceSample
from data_structure.sequence_collection import SequenceCollection

# ----------------------------------------------------
# 1. Utility Function (Pandas-based normalization)
# ----------------------------------------------------
def _normalize_window_features_df(
    X_df: pd.DataFrame,
    ohlc_cols: Tuple[str, str, str, str],
    suffix: str,
    reduce: int,
) -> pd.DataFrame:
    """
    For a single DataFrame window, compute normalized OHLC proportions relative to
    the *last close price* in that window.

    Adds new columns like 'open_prop', 'high_prop', etc.
    """
    if not isinstance(X_df, pd.DataFrame):
        print("⚠️ [DEBUG] Expected DataFrame, got:", type(X_df))
        return X_df

    missing_cols = [c for c in ohlc_cols if c not in X_df.columns]
    if len(missing_cols) == len(ohlc_cols):
        print("⚠️ [DEBUG] No OHLC columns found in DataFrame. Skipping.")
        return X_df

    if "close" not in X_df.columns:
        print("⚠️ [DEBUG] 'close' column missing — skipping normalization.")
        return X_df

    last_close = X_df["close"].iloc[-1]
    if not np.isfinite(last_close) or last_close == 0.0:
        print(f"⚠️ [DEBUG] Invalid last close value ({last_close}). Skipping.")
        return X_df

    print(f"[DEBUG] Normalizing with last close = {last_close}")

    for col in ohlc_cols:
        if col not in X_df.columns:
            continue

        X_df[f"{col}{suffix}"] = X_df[col] / last_close
        if reduce == 1:
            X_df[f"{col}{suffix}"] -= 1.0

        print(f"[DEBUG] Added column '{col}{suffix}' — first 3 values:",
              X_df[f"{col}{suffix}"].head(3).tolist())

    return X_df


# ----------------------------------------------------
# 2. Concrete Add-On (Pandas-based Candle Normalization)
# ----------------------------------------------------
class CandleNormalizationAddOn(BaseAddOn):
    """
    Normalizes each window’s OHLC columns relative to the window’s last close price.
    Adds proportional columns like `open_prop`, `high_prop`, etc.
    """

    on_evaluation_priority = 10

    def __init__(
        self,
        ohlc_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
        suffix: str = "_prop",
        feature_group_key: str = "main",
        target_col: str = "target_price",
        reduce: int = 0,
        verbose: bool = True,
    ):
        self.ohlc_cols = ohlc_cols
        self.suffix = suffix
        self.feature_group_key = feature_group_key
        self.target_col = target_col
        self.reduce = reduce
        self.verbose = verbose

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        if self.verbose:
            print("\n[DEBUG] >>> Entering CandleNormalizationAddOn.apply_window <<<")

        samples_collection: SequenceCollection = state.get("samples")
        if samples_collection is None or len(samples_collection) == 0:
            if self.verbose:
                print("⚠️ [DEBUG] No samples found in state — skipping.")
            return state

        for i, sample in enumerate(samples_collection):
            X_df = sample.X.get(self.feature_group_key)
            if X_df is None:
                if self.verbose:
                    print(f"⚠️ [DEBUG] Sample {i}: Missing feature group '{self.feature_group_key}'.")
                continue

            if not isinstance(X_df, pd.DataFrame):
                if self.verbose:
                    print(f"⚠️ [DEBUG] Sample {i}: Expected DataFrame but got {type(X_df)}.")
                continue

            if self.verbose:
                print(f"[DEBUG] Sample {i}: Original columns = {list(X_df.columns)}")

            X_transformed = _normalize_window_features_df(
                X_df.copy(),
                ohlc_cols=self.ohlc_cols,
                suffix=self.suffix,
                reduce=self.reduce,
            )

            sample.X[self.feature_group_key] = X_transformed

            if self.verbose:
                added_cols = [c for c in X_transformed.columns if c.endswith(self.suffix)]
                print(f"[DEBUG] Sample {i}: Added proportional columns = {added_cols}")

        if self.verbose:
            print("[DEBUG] <<< Exiting CandleNormalizationAddOn.apply_window >>>\n")

        return state

