import importlib
import inspect
def make_step(func, **kwargs):
    """
    Wraps a function and saves metadata for serialization.
    Handles functions with any number of arguments.
    """
    # Create a wrapper function
    def step_fn(*args, **inner_kwargs):
        # Combine the kwargs passed directly and those passed via **kwargs
        final_kwargs = {**inner_kwargs, **kwargs}
        return func(*args, **final_kwargs)

    # Save metadata for the step
    step_fn._step_name = func.__name__
    step_fn._step_module = func.__module__
    step_fn._step_kwargs = kwargs
    return step_fn
