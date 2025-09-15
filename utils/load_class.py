from importlib import import_module
import inspect

def load_class(module_path: str, **kwargs):
    """
    Dynamically import a class/function from a module.
    If a class is found, it will be instantiated with kwargs.
    """
    module = import_module(module_path)

    # If module defines exactly one class
    classes = [cls for _, cls in inspect.getmembers(module, inspect.isclass)
               if cls.__module__ == module.__name__]

    if len(classes) == 1:
        return classes[0](**kwargs)

    # Otherwise assume it defines a factory function "build"
    if hasattr(module, "build"):
        return module.build(**kwargs)

    raise ValueError(f"No suitable class or build() function found in {module_path}")
