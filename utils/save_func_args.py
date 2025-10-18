import inspect

def save_function_args(func, *args, **kwargs):
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments  # <-- returns an OrderedDict