# utils/decorators/run.py

import time
from functools import wraps
from typing import Callable, Dict, Any
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
                    state = hook(state, pipeline_extra_info)

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

            elif mode == "onetime":
                # Logic for sequencer
                if self.sequencer_fn:
                    state = self.sequencer_fn(state, pipeline_extra_info)
            
            return state
        return wrapper
    return decorator