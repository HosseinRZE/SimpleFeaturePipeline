import pandas as pd
from typing import List, Dict, Any
from add_ons.base_addon import BaseAddOn 

class DropColumnsWindowAddOn(BaseAddOn):
    """
    An Add-on to remove specified columns from DataFrames within window lists.
    It now supports targeting specific feature groups (e.g., 'main', 'aux') 
    within the nested dictionary structure of X_list.
    
    The new cols_map format is:
    {
        "X_list": {"main": ["col1", "col2"], "aux": ["col3"]},
        "y_list": {"labels": ["raw_label"]} # For simple list, use a dummy key or just a list as before.
    }
    """
    def __init__(self, cols_map: Dict[str, Any]):
        """
        Initializes the add-on with a map of target lists to columns/groups.

        Args:
            cols_map (Dict[str, Any]): A dictionary where keys are the 
                target list names (e.g., "X_list"). Values can be:
                1. A List[str] for simple lists (e.g., "y_list").
                2. A Dict[str, List[str]] to target specific feature groups 
                   (e.g., {"main": ["cols"]} for "X_list").
        """
        self.cols_map = cols_map

    def apply_window(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Iterates through the target lists and removes columns as specified.
        """
        
        for list_key, drop_spec in self.cols_map.items():
            if list_key not in state or not state[list_key]:
                continue # Skip if the target list is not in state or is empty

            window_list = state[list_key]
            transformed_list = []
            
            # Case 1: X_list (Nested Dictionary structure)
            if list_key == "X_list" and isinstance(drop_spec, dict):
                
                # Check that all samples in X_list are dictionaries
                if not all(isinstance(item, dict) for item in window_list):
                    print(f"🛑 Warning: '{list_key}' specified with group keys, but items are not dictionaries. Skipping.")
                    continue
                
                # drop_spec is now a map like {"main": ["col1", "col2"], "aux": ["col3"]}
                group_map: Dict[str, List[str]] = drop_spec

                for window_item in window_list:
                    new_sample_dict = {}
                    
                    # Iterate through the inner dict (e.g., {'main': df, 'aux': df})
                    for group_key, group_df in window_item.items():
                        
                        # Check if this group_key is specified for dropping columns
                        if group_key in group_map and isinstance(group_df, pd.DataFrame):
                            cols_to_drop = group_map[group_key]
                            cols_to_remove = [c for c in cols_to_drop if c in group_df.columns]
                            
                            # Drop from the inner DataFrame
                            new_sample_dict[group_key] = group_df.drop(columns=cols_to_remove)
                            
                        else:
                            # Keep DataFrames for groups not specified, and keep non-DataFrame data
                            new_sample_dict[group_key] = group_df
                            
                    transformed_list.append(new_sample_dict)
                
                print(f"✅ Dropped columns for '{list_key}' by group: {group_map}")

            # Case 2: Simple List (e.g., a list of DataFrames like y_list)
            elif isinstance(drop_spec, list):
                cols_to_drop: List[str] = drop_spec
                
                for window_item in window_list:
                    if isinstance(window_item, pd.DataFrame):
                        cols_to_remove = [c for c in cols_to_drop if c in window_item.columns]
                        transformed_list.append(window_item.drop(columns=cols_to_remove))
                    else:
                        # Pass through items that aren't DataFrames (like numpy arrays in y_list)
                        transformed_list.append(window_item)
                
                print(f"✅ Dropped columns for '{list_key}': {cols_to_drop}")

            # Case 3: Pass-through for unrecognized spec format
            else:
                print(f"⚠️ Warning: Unrecognized drop specification format for '{list_key}'. Skipping.")
                transformed_list = window_list # Keep the original list

            # Update the state with the modified list
            state[list_key] = transformed_list
            
        return state