from add_ons.base_addon import BaseAddOn 
from typing import List, Dict, Any, Literal
import pandas as pd

# Define the supported target DataFrames in the state
TargetDF = Literal["df_data", "df_labels"]

class DropColumnsAddOn(BaseAddOn):
    """
    An Add-on to remove specified columns from a target DataFrame in the state.
    This operation runs before the data is converted to sequences.

    Examples:
    >>> # 1. Drop feature columns (e.g., 'id', 'ts') from df_data
    >>> drop_features_addon = DropColumnsAddOn(
    ...     cols_to_drop=["id", "ts", "unnecessary_feature"], 
    ...     target_df_key="df_data"
    ... )
    
    >>> # 2. Drop a label column (e.g., 'raw_price') from df_labels
    >>> drop_label_addon = DropColumnsAddOn(
    ...     cols_to_drop=["raw_price"], 
    ...     target_df_key="df_labels"
    ... )
    """
    def __init__(self, cols_to_drop: List[str], target_df_key: TargetDF = "df_data"):
        """
        Initializes the add-on.

        Args:
            cols_to_drop (List[str]): List of column names to remove.
            target_df_key (TargetDF): The key in the 'state' dict holding the target DataFrame 
                                      (e.g., 'df_data' or 'df_labels').
        
        Examples:
        >>> # Initialize to drop 'timestamp' and 'volume' from the main data ('df_data')
        >>> addon = DropColumnsAddOn(cols_to_drop=["timestamp", "volume"], target_df_key="df_data")
        
        >>> # Initialize to drop 'debug_flag' from the labels data ('df_labels')
        >>> addon = DropColumnsAddOn(cols_to_drop=["debug_flag"], target_df_key="df_labels")
        """
        self.cols_to_drop = cols_to_drop
        self.target_df_key = target_df_key

    def before_sequence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Removes the specified columns from the target DataFrame in the state.
        
        Args:
            state (Dict): Contains at least 'df_data' and 'df_labels'.
        Returns:
            Dict: Updated state with the modified DataFrame.

        Examples:
        >>> # Setup a dummy state for testing the logic
        >>> initial_data = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        >>> initial_state = {'df_data': initial_data.copy()}
        
        >>> # Initialize add-on to drop 'b'
        >>> addon = DropColumnsAddOn(cols_to_drop=["b", "d"], target_df_key="df_data")
        
        >>> # Run the stage
        >>> updated_state = addon.before_sequence(initial_state)
        
        >>> # Verify column 'b' is gone and 'd' (which didn't exist) didn't cause an error
        >>> print(updated_state['df_data'].columns.tolist())
        ['a', 'c']
        """
        if self.target_df_key not in state:
            print(f"Warning: Target DataFrame '{self.target_df_key}' not found in state. Skipping column drop.")
            return state

        df = state[self.target_df_key]
        
        # Identify columns present in the DataFrame that should be dropped
        cols_to_remove = [c for c in self.cols_to_drop if c in df.columns]
        
        if cols_to_remove:
            print(f"Dropping columns from {self.target_df_key}: {cols_to_remove}")
            # Use .drop() to remove the columns
            df_out = df.drop(columns=cols_to_remove)
            
            # Update the state with the modified DataFrame
            state[self.target_df_key] = df_out
        else:
            print(f"No columns to drop found in {self.target_df_key} from list: {self.cols_to_drop}")

        return state