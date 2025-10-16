from typing import List, Dict, Any, Tuple

class BaseAddOn:
    """Abstract base class for a pipeline processing step (add-on)."""
    def before_sequence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Operates on the initial, non-sequenced data.
        Typically loads, cleans, or fits scalers on the entire dataset.
        
        Args:
            state (Dict): A dictionary containing at least 'df_data' and 'df_labels'.
            
        Returns:
            Dict: The updated state dictionary.
        """
        return state

    def apply_window(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Operates on the data after it has been split into sequences.
        Applies transformations to each window/sequence.
        
        Args:
            state (Dict): A dictionary containing 'X_list', 'y_list', etc.
            
        Returns:
            Dict: The updated state dictionary.
        """
        return state

    def transformation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies transformations that might change the data's shape (e.g., flattening).
        Runs after apply_window.
        
        Args:
            state (Dict): A dictionary containing 'X_list', 'y_list', etc.
            
        Returns:
            Dict: The updated state dictionary.
        """
        return state