import importlib

def make_step(func, **kwargs):
    """
    Wraps a function and saves metadata for serialization.
    """
    step_fn = lambda df: func(df, **kwargs)
    step_fn._step_name = func.__name__
    step_fn._step_module = func.__module__
    step_fn._step_kwargs = kwargs
    return step_fn
