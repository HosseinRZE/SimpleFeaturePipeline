from typing import List, Dict, Any, Tuple, Callable
from add_ons.base_addon import BaseAddOn
import numpy as np

class FeaturePipeline:
    """Container for pipeline add-ons and shared parameters."""
    
    def __init__(
        self, 
        sequencer: Callable, 
        add_ons: List[BaseAddOn] = None, 
        extra_params: Dict[str, Any] = None
    ):
        self.add_ons = add_ons if add_ons is not None else []
        self.extra_params = extra_params if extra_params is not None else {}
        # 🌟 Store the provided function as the 'sequencer' method's logic
        self._sequencer_logic = sequencer 

    def add(self, addon: BaseAddOn):
        """Register a new add-on to the pipeline."""
        self.add_ons.append(addon)

    def sequencer(self, state: Dict[str, Any]) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray]]:
        """
        Executes the stored sequencer function to generate sequences (X, y).
        The state is updated with sequence lengths internally by the function.
        """
        # 🌟 Call the stored function and pass the state
        X_list, y_list = self._sequencer_logic(state)
        return X_list, y_list