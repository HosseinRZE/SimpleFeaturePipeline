from typing import Dict, Any, Tuple
from add_ons.base_addon import BaseAddOn
import pandas as pd
import numpy as np
from data_structure.sequence_collection import SequenceCollection

class CandleShapeFeaturesAddOn(BaseAddOn):
    """
    Computes candlestick shape features (upper_shadow, lower_shadow, body, color) 
    for each window and adds them to the feature collection.
    """
    
    def __init__(
        self,
        ohlc_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
        feature_group_key: str = "main",
        seperatable: str = "no",
        dict_name: str = "candle_shape",
    ):
        """
        Args:
            ohlc_cols: (open, high, low, close) column names.
            feature_group_key: The existing feature group (dict key in sample.X) to read from and/or write to.
            seperatable: "no" (default) -> add to feature_group_key only (merged)
                         "complete" -> add to dict_name only (new feature group)
                         "both" -> add to both
            dict_name: Name for the new feature group if seperatable != "no".
        """
        self.ohlc_cols = ohlc_cols
        self.feature_group_key = feature_group_key
        self.seperatable = seperatable
        self.dict_name = dict_name
        
        if self.seperatable not in ("no", "complete", "both"):
             raise ValueError(f"Invalid seperatable value: {seperatable}. Must be 'no', 'complete', or 'both'.")

    @staticmethod
    def _compute_candle_features(
        df: pd.DataFrame,
        open_col: str,
        high_col: str,
        low_col: str,
        close_col: str,
    ) -> pd.DataFrame:
        """Core feature computation logic."""
        
        # 1. Determine Max/Min of Open/Close (OC)
        max_oc = df[[open_col, close_col]].max(axis=1)
        min_oc = df[[open_col, close_col]].min(axis=1)
        
        # 2. Create safe denominators (avoid division by zero/NaN)
        # Note: The original logic used max_oc_safe and min_oc_safe only to calculate 
        # invalid_indices, but we'll use a single denominator (max_oc) for normalization.
        denominator = max_oc.replace(0, np.nan) 
        
        # 3. Compute features, using max_oc as the body/total range denominator
        
        # Upper Shadow: (High - Max(O,C)) / Max(O,C)
        upper_shadow = (df[high_col] - max_oc) / denominator
        
        # Lower Shadow: (Min(O,C) - Low) / Max(O,C)
        lower_shadow = (min_oc - df[low_col]) / denominator
        
        # Body: (Max(O,C) - Min(O,C)) / Max(O,C)
        body = (max_oc - min_oc) / denominator
        
        # Color: Simple ternary encoding
        color = np.where(
            df[close_col] > df[open_col], 0.7, # Bullish (Green/Up)
            np.where(df[close_col] < df[open_col], 0.3, 0.0) # Bearish (Red/Down) or Doji/Neutral
        )
        
        # 4. Prepare output DataFrame (NaNs are handled by fillna below)
        features_df = pd.DataFrame({
            "upper_shadow": upper_shadow.fillna(0.0).astype(np.float32),
            "lower_shadow": lower_shadow.fillna(0.0).astype(np.float32),
            "body": body.fillna(0.0).astype(np.float32),
            "color": color.astype(np.float32)
        }, index=df.index)
        
        return features_df

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        
        self.add_trace_print(
            pipeline_extra_info, 
            f"Starting 'apply_window' for CandleShape features (Group='{self.feature_group_key}', Sep='{self.seperatable}')."
        )
        
        samples_collection: SequenceCollection = state.get("samples")
        if not samples_collection:
            self.add_trace_print(pipeline_extra_info, "Skipped: 'samples' collection is empty or missing.")
            return state

        open_col, high_col, low_col, close_col = self.ohlc_cols

        for i, sample in enumerate(samples_collection):
            log_prefix = f"[Sample {i}] "
            
            # 1. Get the target DataFrame (must be mutable)
            X_df = sample.X.get(self.feature_group_key)
            
            if X_df is None or not isinstance(X_df, pd.DataFrame):
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped: Feature group '{self.feature_group_key}' is missing or not a DataFrame."
                )
                continue
            
            # 2. Check for required OHLC columns
            if not all(col in X_df.columns for col in self.ohlc_cols):
                 self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped: Missing one or more OHLC columns {self.ohlc_cols} in feature group."
                )
                 continue

            # 3. Compute Features
            features_df = self._compute_candle_features(
                X_df, open_col, high_col, low_col, close_col
            )
            
            self.add_trace_print(
                pipeline_extra_info, 
                f"{log_prefix}Computed {len(features_df.columns)} candle shape features (Shape: {features_df.shape})."
            )

            # 4. Apply Seperatable Logic
            
            # A. Add to main feature group
            if self.seperatable in ("no", "both"):
                sample.X[self.feature_group_key] = pd.concat([X_df, features_df], axis=1)
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Merged new features into primary group '{self.feature_group_key}'."
                )

            # B. Add to new separate feature group
            if self.seperatable in ("complete", "both"):
                sample.X[self.dict_name] = features_df.copy()
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Created/updated separate feature group '{self.dict_name}'."
                )

        self.add_trace_print(pipeline_extra_info, "Finished 'apply_window' for CandleShape features.")
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference-time hook: applies the same feature generation as during training.
        """
        self.add_trace_print(pipeline_extra_info, "Executing 'on_server_request' (Inference). Delegating to apply_window.")
        return self.apply_window(state, pipeline_extra_info)