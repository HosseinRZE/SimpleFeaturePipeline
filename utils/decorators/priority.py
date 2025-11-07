# In a decorator utils file (e.g., utils/decorators/priority.py)
from functools import wraps
from typing import Callable

def priority(priority_value: int) -> Callable:
    """
    Method decorator for Add-Ons. 
    
    Attaches a priority value to a hook method, which is then
    read by the BaseAddOn.__init__ to set the required 
    instance attribute (e.g., 'on_evaluation_priority').
    """
    def decorator(func: Callable) -> Callable:
        # Just tag the function object with the priority
        func._hook_priority = priority_value
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        # Make sure the wrapper also has the tag
        wrapper._hook_priority = priority_value
        return wrapper
    return decorator