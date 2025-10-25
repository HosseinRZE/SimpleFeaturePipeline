from typing import Dict, Any
import pandas as pd
from add_ons.base_addon import BaseAddOn 
from utils.decorators.track import track
class FeatureNamer(BaseAddOn):
    """
    An add-on to extract the names of all feature columns.
    
    It inspects the first data sample to find the column names across all
    feature groups (if data is in a dict) and stores a flat list of these
    names in state['feature_columns'].
    """
    @track(input='state["X_list"]', output='state["feature_columns"]')
    def transformation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts feature names from the first sample in X_list."""
        X_list = state.get('X_list')

        # If there's no data, there are no feature names.
        if not X_list:
            state['feature_columns'] = []
            return state

        first_sample = X_list[0]
        all_feature_names = []

        if isinstance(first_sample, dict):
            # Case 1: Multi-input, features are in a dictionary of DataFrames.
            # e.g., {'main': df1, 'aux': df2}
            for feature_group_df in first_sample.values():
                if isinstance(feature_group_df, pd.DataFrame):
                    all_feature_names.extend(feature_group_df.columns.tolist())
        
        elif isinstance(first_sample, pd.DataFrame):
            # Case 2: Single-input, features are a single DataFrame.
            all_feature_names = first_sample.columns.tolist()

        # Store the final list of names in the state
        state['feature_columns'] = all_feature_names
        
        print(f"✅ Extracted {len(all_feature_names)} feature names.")
        return state