from typing import List, Dict, Any
from add_ons.base_addon import BaseAddOn
from utils.decorators.run import run


class FeaturePipeline:
    def __init__(self, add_ons: List[BaseAddOn] = None):
        self.add_ons = add_ons if add_ons is not None else []
        self.extra_info: Dict[str, Any] = {}

    # --------------------------------------------------
    # Core pipeline phases
    # --------------------------------------------------

    @run("before_sequence", mode="loop", track=True)
    def run_before_sequence(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs before_sequence hooks on all add-ons."""
        return state

    # --- 🧩 New Step: Sequencing on Training ---
    @run("sequence_on_train", mode="one_time", track=True)
    def run_sequence_on_train(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs sequence_on_train hooks.
        Converts raw DataFrame data into a SequenceCollection during training.
        """
        return state

    # --- 🧩 New Step: Sequencing on Server (Inference) ---
    @run("sequence_on_server", mode="loop", track=True)
    def run_sequence_on_server(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs sequence_on_server hooks.
        Converts live DataFrame data into a SequenceCollection during inference.
        """
        return state

    @run("apply_window", mode="loop", track=True)
    def run_apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs apply_window hooks on all add-ons."""
        return state

    @run("transformation", mode="loop", track=True)
    def run_transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs transformation hooks on all add-ons."""
        return state

    # --- Evaluation Phases ---
    @run("on_evaluation", mode="priority_loop", track=True)
    def run_on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs all on_evaluation hooks in priority order."""
        return eval_data

    @run("on_evaluation_end", mode="priority_loop", track=True)
    def run_on_evaluation_end(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs all on_evaluation_end hooks in priority order."""
        return eval_data

    # --- Server Hooks ---
    @run("on_server_init", mode="loop", track=True)
    def run_on_server_init(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Executed once when the server is initialized."""
        return state

    @run("on_first_request", mode="loop", track=True)
    def run_on_first_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Executed on the first /get_and_add_data request."""
        return state

    @run("on_server_request", mode="loop", track=True)
    def run_on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔹 Server Request Hook 🔹
        Called on each /predict request.

        CONTRACT:
        After all add-ons are executed, `state` MUST contain:
            state["dict_x"]: Dict[str, torch.Tensor]
            state["lengths"]: List[int]
        """
        if "dict_x" not in state or "lengths" not in state:
            raise ValueError(
                "❌ 'run_on_server_request' must produce 'dict_x' and 'lengths' in the returned state."
            )
        return state

    @run("on_server_inference", mode="loop", track=True)
    def run_on_server_inference(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs inference post-processing hooks."""
        return state
    
    @run("on_final_output", mode="one_time", track=True)
    def run_final_output(
        self,
        state: Dict[str, Any],
        pipeline_extra_info: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Runs all addons tagged as final output processors (e.g. PrepareOutputAddOn).
        """

        return state