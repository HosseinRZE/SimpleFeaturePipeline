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
    # New parameters required for logging from a standalone function
    addon_instance: BaseAddOn,
    pipeline_extra_info: Dict[str, Any],
    sample_index: int
) -> pd.DataFrame:
    """Normalize OHLC columns relative to last close price in the window."""
    
    log_prefix = f"[Sample {sample_index}] Helper: "

    if not isinstance(X_df, pd.DataFrame):
        addon_instance.add_trace_print(pipeline_extra_info, f"{log_prefix}Skipped normalization: Input is not a DataFrame.")
        return X_df

    if "close" not in X_df.columns:
        addon_instance.add_trace_print(pipeline_extra_info, f"{log_prefix}Skipped normalization: 'close' column is missing in feature group.")
        return X_df

    last_close = X_df["close"].iloc[-1]

    addon_instance.add_trace_print(pipeline_extra_info, f"{log_prefix}Window shape: {X_df.shape}. Last Close value found: {last_close:.4f}")

    if not np.isfinite(last_close) or last_close == 0.0:
        addon_instance.add_trace_print(pipeline_extra_info, f"{log_prefix}Skipped normalization: Last close ({last_close}) is invalid (NaN, Inf, or Zero).")
        return X_df

    for col in ohlc_cols:
        if col not in X_df.columns:
            addon_instance.add_trace_print(pipeline_extra_info, f"{log_prefix}Column '{col}' not found in DataFrame, skipping.")
            continue
        
        new_col_name = f"{col}{suffix}"
        
        # Perform division (normalization)
        X_df[new_col_name] = X_df[col] / last_close
        
        log_message = f"{log_prefix}Normalized '{col}' to '{new_col_name}'."
        
        # Perform reduction if requested
        if reduce == 1:
            X_df[new_col_name] -= 1.0
            log_message += " Applied reduction (-1.0)."
        
        addon_instance.add_trace_print(pipeline_extra_info, log_message)

    addon_instance.add_trace_print(pipeline_extra_info, f"{log_prefix}Normalization complete for window.")
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
        
        self.add_trace_print(
            pipeline_extra_info, 
            f"Starting 'apply_window' with config: Feature Group='{self.feature_group_key}', OHLC={self.ohlc_cols}, Reduce={self.reduce}."
        )

        samples_collection: SequenceCollection = state.get("samples")
        if not samples_collection:
            self.add_trace_print(pipeline_extra_info, "Skipped: 'samples' collection is empty or missing in state.")
            return state

        for i, sample in enumerate(samples_collection):
            log_prefix = f"[Sample {i}] "
            
            X_df = sample.X.get(self.feature_group_key)
            
            if X_df is None or not isinstance(X_df, pd.DataFrame):
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped sample: Feature group '{self.feature_group_key}' is missing or not a DataFrame."
                )
                continue
            
            self.add_trace_print(
                pipeline_extra_info, 
                f"{log_prefix}Found DataFrame for processing (Shape: {X_df.shape})."
            )
            
            X_transformed = _normalize_window_features_df(
                X_df.copy(),
                ohlc_cols=self.ohlc_cols,
                suffix=self.suffix,
                reduce=self.reduce,
                # Pass logging context for helper function prints
                addon_instance=self,
                pipeline_extra_info=pipeline_extra_info,
                sample_index=i
            )
            
            sample.X[self.feature_group_key] = X_transformed
            self.add_trace_print(pipeline_extra_info, f"{log_prefix}Successfully saved normalized data back to sample X.")

        self.add_trace_print(pipeline_extra_info, "Finished 'apply_window' normalization for all samples.")
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference-time hook: applies the same normalization to `samples` as during training.
        """
        self.add_trace_print(pipeline_extra_info, "Executing 'on_server_request' (Inference). Delegating to apply_window.")
        return self.apply_window(state, pipeline_extra_info)