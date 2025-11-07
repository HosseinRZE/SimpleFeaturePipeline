import time
from functools import wraps
from typing import Callable, Dict, Any, List
# Assuming BaseAddOn is correctly imported and available
from add_ons.base_addon import BaseAddOn 
import operator

# Add 'order' parameter for sorting in method_table
def run(hook_name: str, mode: str = "loop", track: bool = False, order: int = 999):
    """
    Executes an add-on hook on the pipeline and optionally tracks it.

    Args:
        hook_name (str): The name of the add-on method to call (e.g., "apply_window").
        mode (str): Execution mode ("loop", "onetime", "priority_loop").
        track (bool): If True, this step will be logged by the @trace decorator.
        order (int): Defines the display order in `method_table`.
    """
    def decorator(func: Callable):
        
        # --- Attach hook info to the function object ---
        func._pipeline_hook_name = hook_name
        func._pipeline_hook_mode = mode
        func._pipeline_hook_order = order
            
        @wraps(func)
        def wrapper(self, state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            
            # 'self' is the FeaturePipeline instance
            is_tracing = track and hasattr(self, '_trace_log')
            config = getattr(self, '_trace_config', {})
            log = getattr(self, '_trace_log', None)
            
            pipeline_extra_info = kwargs.get('pipeline_extra_info')
            if pipeline_extra_info is None:
                try:
                    pipeline_extra_info = args[0] 
                except IndexError:
                    pipeline_extra_info = {}
            
            # --- Main Execution Logic ---
            if mode in ("loop", "priority_loop"):
                add_on_list = self.add_ons
                
                if mode == "priority_loop":
                    # ... (priority_loop logic) ...
                    priority_attr_name = f"{hook_name}_priority"
                    prioritized_addons: List[Dict[str, Any]] = []
                    for add_on in add_on_list:
                        hook = getattr(add_on, hook_name, None)
                        base_method = getattr(BaseAddOn, hook_name, None)
                        if callable(hook) and hook.__func__ != base_method:
                            if not hasattr(add_on, priority_attr_name):
                                raise ValueError(f"❌ Add-on '{add_on.__class__.__name__}' missing priority attribute '{priority_attr_name}'.")
                            priority = getattr(add_on, priority_attr_name)
                            if not isinstance(priority, (int, float)):
                                raise TypeError(f"❌ Priority attribute '{priority_attr_name}' on '{add_on.__class__.__name__}' must be a number.")
                            prioritized_addons.append({'add_on': add_on, 'hook': hook, 'priority': priority})
                    prioritized_addons.sort(key=operator.itemgetter('priority'), reverse=False)
                    execution_list = prioritized_addons
                
                else: # mode == "loop"
                    execution_list = [{'add_on': add_on, 'hook': getattr(add_on, hook_name, None), 'priority': None} 
                                      for add_on in add_on_list]

                for item in execution_list:
                    add_on = item['add_on']
                    hook = item['hook']
                    
                    if mode == "loop":
                        if not callable(hook):
                            continue
                        base_method = getattr(BaseAddOn, hook_name, None)
                        if hook.__func__ == base_method:
                            continue
                    
                    start_time = 0
                    if is_tracing and config.get('time_track'):
                        start_time = time.perf_counter()

                    state = hook(state, pipeline_extra_info, **kwargs) 

                    if is_tracing:
                        elapsed_time = 0
                        if config.get('time_track'):
                            elapsed_time = time.perf_counter() - start_time
                        
                        message = pipeline_extra_info.pop('current_trace_message', '')
                        
                        trace_data = {
                            'method': func.__name__,
                            'addon': add_on.__class__.__name__,
                            'message': message,
                            'time': elapsed_time
                        }
                        if mode == "priority_loop":
                                trace_data['priority'] = item['priority']
                            
                        log.append(trace_data)

            elif mode == "one_time":
                implementing_addons: List[BaseAddOn] = []
                for add_on in self.add_ons:
                    hook = getattr(add_on, hook_name, None)
                    base_method = getattr(BaseAddOn, hook_name, None)
                    if callable(hook) and hook.__func__ != base_method:
                        implementing_addons.append(add_on)

                if len(implementing_addons) == 0:
                    return state 
                elif len(implementing_addons) > 1:
                    addon_names = [a.__class__.__name__ for a in implementing_addons]
                    raise ValueError(f"❌ '{hook_name}' is 'onetime', but implemented by: {', '.join(addon_names)}.")
                
                add_on = implementing_addons[0]
                hook = getattr(add_on, hook_name)
                
                start_time = 0
                if is_tracing and config.get('time_track'):
                    start_time = time.perf_counter()

                state = hook(state, pipeline_extra_info, **kwargs)

                if is_tracing:
                    elapsed_time = 0
                    if config.get('time_track'):
                        elapsed_time = time.perf_counter() - start_time
                    
                    message = pipeline_extra_info.pop('current_trace_message', '')
                    
                    log.append({
                        'method': func.__name__,
                        'addon': add_on.__class__.__name__,
                        'message': message,
                        'time': elapsed_time
                    })
            else:
                valid_modes = ["loop", "priority_loop", "one_time"]
                raise ValueError(f"❌ Invalid run mode: '{mode}'. Must be one of: {', '.join(valid_modes)}.")
            return state
        
        wrapper._pipeline_hook_name = hook_name
        wrapper._pipeline_hook_mode = mode
        wrapper._pipeline_hook_order = order
        
        return wrapper
    return decorator