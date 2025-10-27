# utils/decorators/run.py

import time
from functools import wraps
from typing import Callable, Dict, Any, List
from add_ons.base_addon import BaseAddOn # Import your BaseAddOn

def run(hook_name: str, mode: str = "loop", track: bool = False):
    """
    Executes an add-on hook on the pipeline and optionally tracks it.

    Args:
        hook_name (str): The name of the add-on method to call (e.g., "apply_window").
        mode (str): Execution mode ("loop", "onetime", "priority_loop").
        track (bool): If True, this step will be logged by the @trace decorator.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            
            # 'self' is the FeaturePipeline instance
            # The @trace decorator must have set up these attributes
            is_tracing = track and hasattr(self, '_trace_log')
            config = getattr(self, '_trace_config', {})
            log = getattr(self, '_trace_log', None)
            
            # The original method signature is (self, state, pipeline_extra_info)
            # We must find pipeline_extra_info in the args or kwargs
            pipeline_extra_info = kwargs.get('pipeline_extra_info')
            if pipeline_extra_info is None:
                try:
                    # Assuming it's the second positional arg after state
                    pipeline_extra_info = args[0] 
                except IndexError:
                    # Fallback for methods without pipeline_extra_info (like sequencer)
                    pipeline_extra_info = {}

            # --- Main Execution Logic ---
            if mode == "loop" or mode == "priority_loop":
                
                # TODO: Implement priority sorting if mode == "priority_loop"
                add_on_list = self.add_ons

                for add_on in add_on_list:
                    hook = getattr(add_on, hook_name, None)
                    if not callable(hook):
                        continue
                    
                    # --- Point 3: Skip default methods ---
                    base_method = getattr(BaseAddOn, hook_name, None)
                    if hook.__func__ == base_method:
                        continue
                    
                    # --- Point 1: Optional Tracking ---
                    start_time = 0
                    if is_tracing and config.get('time_track'):
                        start_time = time.perf_counter()

                    # --- Point 2: Call with correct signature ---
                    state = hook(state, pipeline_extra_info, **kwargs)

                    if is_tracing:
                        elapsed_time = 0
                        if config.get('time_track'):
                            elapsed_time = time.perf_counter() - start_time
                        
                        # Read message from the message bus
                        message = pipeline_extra_info.pop('current_trace_message', '')
                        
                        log.append({
                            'method': func.__name__,
                            'addon': add_on.__class__.__name__,
                            'message': message,
                            'time': elapsed_time
                        })

            elif mode == "one_time":
                # --- New Logic for Onetime Execution ---
                
                # 1. Collect all implementing add-ons
                implementing_addons: List[BaseAddOn] = []
                for add_on in self.add_ons:
                    hook = getattr(add_on, hook_name, None)
                    base_method = getattr(BaseAddOn, hook_name, None)
                    
                    if callable(hook) and hook.__func__ != base_method:
                        implementing_addons.append(add_on)

                # 2. Enforce the "one-time" constraint
                if len(implementing_addons) == 0:
                    # No error if no add-on implements an optional one-time step, just pass
                    return state 
                elif len(implementing_addons) > 1:
                    addon_names = [a.__class__.__name__ for a in implementing_addons]
                    raise ValueError(
                        f"❌ '{hook_name}' is a 'onetime' hook, but it was implemented "
                        f"by more than one add-on: {', '.join(addon_names)}. "
                        f"Only one add-on is permitted for this step."
                    )
                
                # 3. Execute the single hook
                add_on = implementing_addons[0]
                hook = getattr(add_on, hook_name) # Already checked to be callable and custom
                
                start_time = 0
                if is_tracing and config.get('time_track'):
                    start_time = time.perf_counter()

                state = hook(state, pipeline_extra_info, **kwargs) # Call with correct signature

                # 4. Handle tracing for the single execution
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
                # Add an explicit error for unhandled modes
                valid_modes = ["loop", "priority_loop", "one_time"]
                raise ValueError(
                    f"❌ Invalid run mode: '{mode}'. Must be one of: {', '.join(valid_modes)}."
                )
            return state
        return wrapper
    return decorator
