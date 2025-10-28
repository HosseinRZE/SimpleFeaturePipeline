import time
from typing import List, Dict, Any
from tabulate import tabulate
import operator  # <-- ADDED IMPORT
from add_ons.base_addon import BaseAddOn
from utils.decorators.run import run
class FeaturePipeline:
    def __init__(self, add_ons: List[BaseAddOn] = None):
        self.add_ons = add_ons if add_ons is not None else []
        self.extra_info: Dict[str, Any] = {}
    # --------------------------------------------------
    # Core pipeline phases (Added 'order=...')
    # --------------------------------------------------
    @run("before_sequence", mode="loop", track=True, order=10)
    def run_before_sequence(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs before_sequence hooks on all add-ons."""
        return state

    @run("sequence_on_train", mode="one_time", track=True, order=20)
    def run_sequence_on_train(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """..."""
        return state

    @run("sequence_on_server", mode="loop", track=True, order=30)
    def run_sequence_on_server(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """..."""
        return state

    @run("apply_window", mode="loop", track=True, order=40)
    def run_apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs apply_window hooks on all add-ons."""
        return state

    @run("transformation", mode="loop", track=True, order=50)
    def run_transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs transformation hooks on all add-ons."""
        return state

    @run("on_evaluation", mode="priority_loop", track=True, order=60)
    def run_on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs all on_evaluation hooks in priority order."""
        return eval_data

    @run("on_evaluation_end", mode="priority_loop", track=True, order=70)
    def run_on_evaluation_end(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs all on_evaluation_end hooks in priority order."""
        return eval_data

    @run("on_server_init", mode="loop", track=True, order=100)
    def run_on_server_init(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Executed once when the server is initialized."""
        return state

    @run("on_first_request", mode="loop", track=True, order=110)
    def run_on_first_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Executed on the first /get_and_add_data request."""
        return state

    @run("on_server_request", mode="loop", track=True, order=120)
    def run_on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """..."""
        if "dict_x" not in state or "lengths" not in state:
            raise ValueError(
                "❌ 'run_on_server_request' must produce 'dict_x' and 'lengths' in the returned state."
            )
        return state

    @run("on_server_inference", mode="loop", track=True, order=130)
    def run_on_server_inference(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """Runs inference post-processing hooks."""
        return state
    
    @run("on_final_output", mode="one_time", track=True, order=140)
    def run_final_output(
        self,
        state: Dict[str, Any],
        pipeline_extra_info: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """..."""
        return state
    # --------------------------------------------------
    # ✨ REBUILT DYNAMIC METHOD ✨
    # --------------------------------------------------
    def method_table(self, show: bool = True) -> str:
        """
        Generates and optionally prints a table of pipeline steps 
        and the add-ons that implement them.

        This method dynamically inspects all methods on this instance,
        finds those decorated with @run, and reads their hook info.

        Args:
            show (bool): If True (default), prints the table to the console.

        Returns:
            str: The formatted table as a string.
        """
        
        headers = ["Order", "Pipeline Method", "Hook Name", "Mode", "Implementing Add-Ons"]
        collected_methods = []

        # Iterate over all members of the class
        for method_name in dir(self):
            if method_name.startswith('_'):
                continue  # Skip private/magic methods

            method = getattr(self, method_name)
            
            # Check if it's a decorated pipeline step by looking for
            # the attribute we added in the @run decorator.
            if not callable(method) or not hasattr(method, '_pipeline_hook_name'):
                continue

            # It is! Get the info.
            hook_name = method._pipeline_hook_name
            mode = method._pipeline_hook_mode
            order = method._pipeline_hook_order
            
            # Find all add-ons that implement this specific hook
            implementing_addons = []
            for add_on in self.add_ons:
                hook = getattr(add_on, hook_name, None)
                base_method = getattr(BaseAddOn, hook_name, None)
                
                # Check if the add-on has a *custom* implementation
                if callable(hook) and hook.__func__ != base_method:
                    implementing_addons.append(add_on.__class__.__name__)
            
            addons_str = ", ".join(implementing_addons) if implementing_addons else "---"
            
            collected_methods.append({
                'order': order,
                'method_name': method_name,
                'hook_name': hook_name,
                'mode': mode,
                'addons_str': addons_str
            })
        
        # Sort the collected methods by the 'order' we defined
        collected_methods.sort(key=operator.itemgetter('order'))
        
        # Build the final table data from the sorted list
        table_data = [
            [
                item['order'], 
                item['method_name'], 
                item['hook_name'], 
                item['mode'], 
                item['addons_str']
            ] 
            for item in collected_methods
        ]

        table = tabulate(table_data, headers=headers, tablefmt="fancy_grid")
        
        if show:
            header_text = f"--- Add-On Hook Implementation for Pipeline ---"
            print("\n" + "=" * len(header_text))
            print(header_text)
            print(table)
            print("=" * len(header_text) + "\n")
        
        return table