import inspect

def get_init_args(obj, **overrides):
    """
    Return all constructor arguments of obj, filled with given overrides or default values.
    """
    cls = obj.__class__
    sig = inspect.signature(cls.__init__)
    args = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if name in overrides:
            args[name] = overrides[name]
        elif param.default is not inspect.Parameter.empty:
            args[name] = param.default
        else:
            raise ValueError(f"Parameter {name} has no default and is not in overrides")
    return args