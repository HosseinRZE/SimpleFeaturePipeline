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

    @run("on_server_request", mode="loop", track=True, order=120)
    def run_on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """..."""
        if "dict_x" not in state or "lengths" not in state:
            raise ValueError(
                "❌ 'run_on_server_request' must produce 'dict_x' and 'lengths' in the returned state."
            )
        return state

    @run("on_server_inference", mode="priority_loop", track=True, order=130)
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
            and the add-ons that implement them, IN THE CORRECT EXECUTION ORDER.

            This method dynamically inspects all methods on this instance,
            finds those decorated with @run, and reads their hook info.
            It now correctly sorts add-ons for 'priority_loop' mode.

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
                
                # Check if it's a decorated pipeline step
                if not callable(method) or not hasattr(method, '_pipeline_hook_name'):
                    continue

                # It is! Get the info.
                hook_name = method._pipeline_hook_name
                mode = method._pipeline_hook_mode
                order = method._pipeline_hook_order
                
                # Find all add-ons that implement this specific hook
                implementing_addons_list = []
                
                # --- MODIFIED LOGIC ---
                # Replicate the execution logic based on the mode
                
                if mode == "priority_loop":
                    # --- NEW: Logic copied from @run decorator ---
                    priority_attr_name = f"{hook_name}_priority"
                    prioritized_addons: List[Dict[str, Any]] = []
                    
                    for add_on in self.add_ons:
                        hook = getattr(add_on, hook_name, None)
                        base_method = getattr(BaseAddOn, hook_name, None)
                        
                        if callable(hook) and hook.__func__ != base_method:
                            # We don't need to raise errors here, just check
                            if hasattr(add_on, priority_attr_name):
                                priority = getattr(add_on, priority_attr_name)
                                prioritized_addons.append({
                                    'add_on': add_on, 
                                    'priority': priority if isinstance(priority, (int, float)) else 999
                                })
                            else:
                                # Add-on implements it but has no priority? Put it last.
                                prioritized_addons.append({'add_on': add_on, 'priority': 9999}) 
                    
                    # Sort them just like the decorator does
                    prioritized_addons.sort(key=operator.itemgetter('priority'), reverse=False)
                    
                    # Format the name string to include priority
                    implementing_addons_list = [
                        f"{item['add_on'].__class__.__name__} (P:{item['priority']})" 
                        for item in prioritized_addons
                    ]
                    # --- End new logic ---

                else: # "loop" or "one_time"
                    # --- This is the ORIGINAL logic ---
                    for add_on in self.add_ons:
                        hook = getattr(add_on, hook_name, None)
                        base_method = getattr(BaseAddOn, hook_name, None)
                        
                        if callable(hook) and hook.__func__ != base_method:
                            implementing_addons_list.append(add_on.__class__.__name__)
                
                # --- MODIFIED ---
                addons_str = ", ".join(implementing_addons_list) if implementing_addons_list else "---"
                
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