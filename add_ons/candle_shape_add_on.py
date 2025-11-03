from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
class CandleNormalizationAddOn(BaseAddOn):
    """
    Normalizes each window’s OHLC columns relative to the window’s last close price.
    
    If `separate_dict=False`, new normalized columns are added to the 'main' feature group.
    If `separate_dict=True`, a new feature group 'candle_shape' is created containing only
    the normalized columns.
    """
    def __init__(
        self,
        ohlc_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
        suffix: str = "_prop",
        feature_group_key: str = "main",
        separate_dict: bool = False,
    ):
        self.ohlc_cols = ohlc_cols
        self.suffix = suffix
        self.feature_group_key = feature_group_key
        self.separate_dict = separate_dict

    def _normalize_ohlc_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLC columns relative to last close price in the window."""
        if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
            return df

        df_out = df.copy()
        last_close = df_out["close"].iloc[-1]
        if not np.isfinite(last_close) or last_close == 0.0:
            return df_out

        norm_cols = {}
        for col in self.ohlc_cols:
            if col not in df_out.columns:
                continue
            norm_cols[f"{col}{self.suffix}"] = df_out[col] / last_close

        # Return a DataFrame with normalized columns only
        norm_df = pd.DataFrame(norm_cols, index=df_out.index, dtype=np.float32)
        return norm_df

    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Applies normalization to each sample in the 'samples' SequenceCollection."""
        samples_collection: SequenceCollection = state.get("samples")
        if not samples_collection:
            return state

        for sample in samples_collection:
            X_df = sample.X.get(self.feature_group_key)
            if X_df is None or not isinstance(X_df, pd.DataFrame):
                continue

            normalized_df = self._normalize_ohlc_features_df(X_df)

            if self.separate_dict:
                # Create new feature group
                sample.X["candle_shape"] = normalized_df
            else:
                # Add normalized columns into the same feature group
                combined = pd.concat([X_df, normalized_df], axis=1)
                sample.X[self.feature_group_key] = combined

        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Applies the same normalization logic during inference."""
        return self.transformation(state, pipeline_extra_info)
