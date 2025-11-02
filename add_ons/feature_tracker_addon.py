from typing import Dict, Any
from add_ons.base_addon import BaseAddOn 
from data_structure.sequence_collection import SequenceCollection
import pandas as pd

class FeatureColumnTrackerAddOn(BaseAddOn):
    """
    Ensures the global 'feature_cols' in the state is a DICT mapping 
    feature group key to its list of columns.

    This AddOn runs late in the pipeline (high priority) to inspect the 
    DataFrame columns of the first SequenceSample after all prior AddOns 
    (which may have added new features like '_prop') have completed.
    """
    # Set a high priority to run after all feature-modifying AddOns (e.g., CandleNormalizationAddOn)
    apply_window_priority = 90
    
    def __init__(self, feature_group_key: str = "main"):
        """
        Initializes the tracker with the feature group key to inspect.
        
        Args:
            feature_group_key (str): The key in SequenceSample.X (e.g., 'main') 
                                     to inspect for columns.
        """
        self.feature_group_key = feature_group_key

    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspects the features of the first sample and sets the global 'feature_columns'
        to a dictionary mapping the feature group key to the final column list.
        """
        samples_collection: SequenceCollection = state.get('samples')
        
        if not samples_collection or len(samples_collection) == 0:
            return state

        # Get the first sample, which represents the final column schema after all transformations
        first_sample = samples_collection.get_list()[0]
        X_df: pd.DataFrame = first_sample.X.get(self.feature_group_key)
        
        if X_df is not None:
            # --- FIX APPLIED HERE ---
            # Set the global feature_columns to a dict: {key: [column_names]}
            # This matches the expected format for 'run_debug_mode'
            state['feature_columns'] = {
                self.feature_group_key: X_df.columns.tolist()
            }
            # --- END OF FIX ---
            
        return state