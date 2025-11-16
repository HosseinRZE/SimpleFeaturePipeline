from typing import Dict, Any, Union
from add_ons.base_addon import BaseAddOn
import pandas as pd
import numpy as np
from data_structure.sequence_collection import SequenceCollection

class LogReturnAddOn(BaseAddOn):
    """
    Calculates the log return globally on state['df_data'] (the full DataFrame) 
    and applies the sliced results to each window, eliminating per-window NaNs.
    """
    on_evaluation_priority = 10 
    
    def __init__(
        self,
        ohlc_cols: tuple = ("open", "high", "low", "close"),
        prefix: str = "log_return_",
        feature_group_key: str = "main",
        seperatable: bool = False,
        new_feature_group_key: str = "log_returns",
    ):
        super().__init__()
        self.ohlc_cols = ohlc_cols
        self.prefix = prefix
        self.feature_group_key = feature_group_key
        self.seperatable = seperatable
        self.new_feature_group_key = new_feature_group_key

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        
        target_group = self.feature_group_key
        dest_group = self.new_feature_group_key if self.seperatable else self.feature_group_key
        
        self.add_trace_print(
            pipeline_extra_info, 
            "Starting Log Return calculation globally on state['df_data'] (Full DataFrame)."
        )

        # 1. Access the FULL DataFrame directly from state['df_data']
        full_df: pd.DataFrame = state.get('df_data')
        
        if full_df is None or not isinstance(full_df, pd.DataFrame):
            self.add_trace_print(pipeline_extra_info, "Skipped: state['df_data'] is missing or not a DataFrame.")
            return state

        # 2. Calculate Log Returns GLOBALLY
        global_log_returns = pd.DataFrame(index=full_df.index)
        
        for col in self.ohlc_cols:
            if col not in full_df.columns:
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"Column '{col}' not found in global df, skipping."
                )
                continue
                
            new_col_name = f"{self.prefix}{col}"
            
            # Global Calculation: log(P_t) - log(P_{t-1})
            # We use .loc[:, col] to ensure we're slicing the intended column data structure
            log_returns = np.log(full_df.loc[:, col]).diff(1) 
            global_log_returns[new_col_name] = log_returns
            
            # --- DEBUG: Show results of global calculation ---
            print(f"[DEBUG: GLOBAL LOG RETURN] Calculated '{new_col_name}'. NaNs: {log_returns.isnull().sum()}. First 3 rows:\n{log_returns.head(3).to_string()}")
            self.add_trace_print(pipeline_extra_info, f"Global log returns calculated for '{new_col_name}'.")

        # 3. Apply Sliced Log Returns to each sample window
        samples_collection: SequenceCollection = state.get("samples")
        if not samples_collection:
            self.add_trace_print(pipeline_extra_info, "Skipped: 'samples' collection is empty or missing.")
            return state

        for i, sample in enumerate(samples_collection):
            log_prefix = f"[Sample {i}] "
            
            # Use the index of the sample's target feature group to slice the global results
            sample_X_data = sample.X.get(target_group)
            
            if sample_X_data is None:
                self.add_trace_print(pipeline_extra_info, f"{log_prefix}Skipped sample: Target group '{target_group}' not found in sample.X.")
                continue

            # Ensure we get the correct index, assuming sample.X[target_group] is a DataFrame/Series
            window_index = sample_X_data.index 
            
            # Slice the global log returns using the window's index
            sliced_log_returns = global_log_returns.loc[window_index]

            if sliced_log_returns.empty:
                 self.add_trace_print(pipeline_extra_info, f"{log_prefix}Skipped: Sliced log returns for window are empty.")
                 continue

            if self.seperatable:
                # Add the new features to a separate dictionary key
                sample.X[dest_group] = sliced_log_returns
                log_message = f"{log_prefix}Saved sliced log returns to a separate feature group: '{dest_group}'."
            else:
                # Merge the new features into the existing feature group
                X_df = sample_X_data.copy()
                X_modified = pd.concat([X_df, sliced_log_returns], axis=1)
                sample.X[dest_group] = X_modified
                log_message = f"{log_prefix}Merged sliced log returns into existing feature group: '{dest_group}'."
            
            self.add_trace_print(pipeline_extra_info, log_message)

        self.add_trace_print(pipeline_extra_info, "Finished Log Return application for all samples.")
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        self.add_trace_print(pipeline_extra_info, "Executing 'on_server_request' (Inference). Delegating to apply_window.")
        return self.apply_window(state, pipeline_extra_info)