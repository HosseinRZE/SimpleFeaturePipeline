from typing import List, Dict, Any, Tuple

class BaseAddOn:
    def __init__(self, *args, **kwargs):

        """
            Base constructor.
            
            ✨ NEW: This __init__ now automatically detects methods
            decorated with @priority(N) and creates the corresponding
            instance attribute (e.g., self.on_evaluation_priority = N).
        """
        for method_name in dir(self):
            if method_name.startswith('_'):
                continue
            
            method = getattr(self, method_name)
            
            # Check if it's a callable method with our magic tag
            if callable(method) and hasattr(method, '_hook_priority'):
                
                # e.g., method_name = "on_evaluation"
                # e.g., priority_val = 1
                priority_val = method._hook_priority
                
                # e.g., attr_name = "on_evaluation_priority"
                attr_name = f"{method_name}_priority"
                
                # e.g., self.on_evaluation_priority = 1
                setattr(self, attr_name, priority_val)
                
    def _set_trace_message(self, pipeline_extra_info: Dict[str, Any], message: str) -> None:
        """Sets the current trace message in the pipeline_extra_info dictionary."""
        if pipeline_extra_info is not None:
            pipeline_extra_info['current_trace_message'] = message
        else:
            # Note: In a real system, you might want a more robust way to handle this warning
            print("Warning: _set_trace_message called but pipeline_extra_info is missing.")

    def add_trace_print(self, pipeline_extra_info: Dict[str, Any], message: str) -> None:
        """
        Adds a message to the pipeline trace. 
        All attachment-related logic has been removed, so this simply calls _set_trace_message.
        """
        self._set_trace_message(pipeline_extra_info, message)