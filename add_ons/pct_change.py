import pandas as pd
import numpy as np
from typing import Dict, Any, List

from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
from data_structure.sequence_sample import SequenceSample

class PctChangeMapperAddOn(BaseAddOn):
    """
    Calculates OHLC percentage changes relative to the previous time step 
    and adds them to the feature collection.
    
    Uses the 'seperatable' logic to determine where to store the new features.
    """
    
    def __init__(
        self, 
        feature_group_key: str = "main",
        seperatable: str = "complete", # Default to creating a new group
        dict_name: str = "pct_changes",
        relative_to: str = "same",  
        features_to_calculate: List[str] = ["open", "high", "low", "close"]
    ):
        """
        Args:
            feature_group_key: The existing feature group (dict key in sample.X) 
                               to read from and/or write to.
            seperatable: "no" -> add to feature_group_key only (merged)
                         "complete" (default) -> add to dict_name only (new feature group)
                         "both" -> add to both
            dict_name: Name for the new feature group if seperatable != "no".
            relative_to: 'same' (pct_change) or 'close' (change relative to prev close).
            features_to_calculate: List of columns to apply pct change to.
        """
        self.feature_group_key = feature_group_key
        self.seperatable = seperatable
        self.dict_name = dict_name
        self.relative_to = relative_to.lower()
        self.features_to_calculate = features_to_calculate

        if self.seperatable not in ("no", "complete", "both"):
            raise ValueError(f"Invalid seperatable value: {seperatable}. Must be 'no', 'complete', or 'both'.")

        if self.relative_to not in ["same", "close"]:
            raise ValueError("relative_to must be either 'same' or 'close'")

        if self.seperatable == "no":
            self.add_trace_print(None, f"PctChangeAddOn: New features will be merged into '{self.feature_group_key}'.")

    def _calculate_pct_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the core calculation logic to a single DataFrame.
        """
        # Check for source columns needed for calculation
        required_cols = self.features_to_calculate.copy()
        if self.relative_to == "close" and "close" not in required_cols:
            required_cols.append("close")
            
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Source group DataFrame is missing required columns: {missing}")

        df_out = pd.DataFrame(index=df.index)
        
        if self.relative_to == "same":
            for col in self.features_to_calculate:
                df_out[f"{col}_pct"] = df[col].pct_change().fillna(0)
        
        elif self.relative_to == "close":
            prev_close = df["close"].shift(1)
            for col in self.features_to_calculate:
                df_out[f"{col}_pct"] = ((df[col] - prev_close) / prev_close).fillna(0)

        # Clean up potential Infs from 0/0 divisions
        return df_out.replace([np.inf, -np.inf], 0).astype(np.float32)

    def _process_samples(self, samples: SequenceCollection, pipeline_extra_info: Dict[str, Any]):
        """
        Iterates through samples and applies the calculation to each one.
        """
        
        for i, sample in enumerate(samples):
            log_prefix = f"[Sample {i}] "
            
            # 1. Get the target DataFrame
            source_data = sample.X.get(self.feature_group_key)
            
            if not isinstance(source_data, pd.DataFrame):
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Skipped: Feature group '{self.feature_group_key}' is missing or not a DataFrame."
                )
                continue

            try:
                # 2. Compute Features
                pct_df = self._calculate_pct_changes(source_data)
                
                self.add_trace_print(
                    pipeline_extra_info, 
                    f"{log_prefix}Computed {len(pct_df.columns)} PctChange features (Shape: {pct_df.shape})."
                )

                # 3. Apply Seperatable Logic
                
                # A. Add to main feature group
                if self.seperatable in ("no", "both"):
                    sample.X[self.feature_group_key] = pd.concat([source_data, pct_df], axis=1)
                    self.add_trace_print(
                        pipeline_extra_info, 
                        f"{log_prefix}Merged new features into primary group '{self.feature_group_key}'."
                    )

                # B. Add to new separate feature group
                if self.seperatable in ("complete", "both"):
                    sample.X[self.dict_name] = pct_df.copy()
                    self.add_trace_print(
                        pipeline_extra_info, 
                        f"{log_prefix}Created/updated separate feature group '{self.dict_name}'."
                    )

            except Exception as e:
                self.add_trace_print(pipeline_extra_info, f"🔥 {log_prefix}Error calculating PctChanges: {e}")

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calculation during the training pipeline (Fit & Transform)."""
        
        self.add_trace_print(
            pipeline_extra_info, 
            f"Starting 'apply_window' for PctChanges (Group='{self.feature_group_key}', Sep='{self.seperatable}')."
        )
        
        samples: SequenceCollection = state.get("samples")
        if not samples:
            self.add_trace_print(pipeline_extra_info, "No samples found; skipping PctChange calculation.")
            return state

        self._process_samples(samples, pipeline_extra_info)
        
        self.add_trace_print(pipeline_extra_info, "Finished 'apply_window' for PctChanges.")
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calculation during server inference (Transform only)."""
        
        # This transformation is stateless (no fitting required)
        self.add_trace_print(pipeline_extra_info, "Executing 'on_server_request' (Inference). Delegating to apply_window.")
        
        return self.apply_window(state, pipeline_extra_info)