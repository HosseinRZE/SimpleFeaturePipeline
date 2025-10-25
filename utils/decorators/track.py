
import time
from functools import wraps
from typing import Callable, Any, Dict, List

def track(func: Callable):
    """
    Makes a FeaturePipeline method trackable by the @trace decorator.
    It loops through add-ons, times them (if configured), and records the results.
    """
    @wraps(func)
    def wrapper(self, state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        # If tracing isn't active, just run the original logic (which is now empty)
        # and return the state. This makes the decorator safe to use without @trace.
        if not hasattr(self, '_trace_log'):
            return state

        method_name = func.__name__
        hook_name = method_name.replace("run_", "") # e.g., "run_apply_window" -> "apply_window"

        for add_on in self.add_ons:
            hook = getattr(add_on, hook_name, None)
            if not callable(hook):
                continue

            start_time = 0
            if self._trace_config.get('time_track'):
                start_time = time.perf_counter()

            # The hook is now passed the pipeline instance `self`
            state = hook(state, self)

            elapsed_time = 0
            if self._trace_config.get('time_track'):
                elapsed_time = time.perf_counter() - start_time
            
            # Retrieve and clear the message for this specific step
            message = self.extra_info.pop('current_trace_message', '')

            self._trace_log.append({
                'method': method_name,
                'addon': add_on.__class__.__name__,
                'message': message,
                'time': elapsed_time
            })
        
        # The original function body is effectively replaced by this loop
        return state
    return wrapper