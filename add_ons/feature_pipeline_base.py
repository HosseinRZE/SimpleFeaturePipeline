from typing import List, Dict, Any, Tuple, Callable
from add_ons.base_addon import BaseAddOn # Assuming BaseAddOn exists and methods return state
from utils.decorators.run import run
from utils.decorators.track import track
class FeaturePipeline:
    def __init__(self, add_ons: List[BaseAddOn] = None ,sequencer: Callable = None):
        self.add_ons = add_ons if add_ons is not None else []
        self.sequencer_fn = sequencer
        # These are set temporarily by the trace decorator
        self.extra_info: Dict[str, Any] = {}
    @run("before_sequence", mode="loop", track=True)
    def run_before_sequence(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs before_sequence hooks on all add-ons."""
        return state

    @run("apply_window", mode="loop", track=True)
    def run_apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs apply_window hooks on all add-ons."""
        return state

    @run("transformation", mode="loop", track=True)
    def run_transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs transformation hooks on all add-ons."""
        return state

    # Not tracked
    @run("sequencer", mode="onetime", track=False)
    def sequencer(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]):
        """Runs the main sequencer function once."""
        return state

    # Tracked
    @run("on_evaluation", mode="priority_loop", track=True)
    def run_on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs all on_evaluation hooks in priority order."""
        return eval_data
    
        # Tracked
    @run("on_evaluation_end", mode="priority_loop", track=True)
    def run_on_evaluation_end(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs all on_evaluation hooks in priority order."""
        return eval_data