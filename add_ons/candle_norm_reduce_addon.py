from typing import Dict, Any
from add_ons.base_addon import BaseAddOn
import pandas as pd
import numpy as np
from data_structure.sequence_collection import SequenceCollection

def _normalize_window_features_df(
    X_df: pd.DataFrame,
    ohlc_cols: tuple,
    suffix: str,
    reduce: int,
) -> pd.DataFrame:
    """Normalize OHLC columns relative to last close price in the window."""
    if not isinstance(X_df, pd.DataFrame):
        return X_df

    if "close" not in X_df.columns:
        return X_df

    last_close = X_df["close"].iloc[-1]
    if not np.isfinite(last_close) or last_close == 0.0:
        return X_df

    for col in ohlc_cols:
        if col not in X_df.columns:
            continue
        X_df[f"{col}{suffix}"] = X_df[col] / last_close
        if reduce == 1:
            X_df[f"{col}{suffix}"] -= 1.0
    return X_df


class CandleNormalizationAddOn(BaseAddOn):
    """
    Normalizes each window’s OHLC columns relative to the window’s last close price.
    Works on both training and server by modifying `sample.X` in place.
    """
    on_evaluation_priority = 10

    def __init__(
        self,
        ohlc_cols: tuple = ("open", "high", "low", "close"),
        suffix: str = "_prop",
        feature_group_key: str = "main",
        reduce: int = 0,
    ):
        self.ohlc_cols = ohlc_cols
        self.suffix = suffix
        self.feature_group_key = feature_group_key
        self.reduce = reduce

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        samples_collection: SequenceCollection = state.get("samples")
        if not samples_collection:
            return state

        for sample in samples_collection:
            X_df = sample.X.get(self.feature_group_key)
            if X_df is None or not isinstance(X_df, pd.DataFrame):
                continue
            X_transformed = _normalize_window_features_df(
                X_df.copy(),
                ohlc_cols=self.ohlc_cols,
                suffix=self.suffix,
                reduce=self.reduce,
            )
            sample.X[self.feature_group_key] = X_transformed

        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference-time hook: applies the same normalization to `samples` as during training.
        """
        return self.apply_window(state, pipeline_extra_info)
