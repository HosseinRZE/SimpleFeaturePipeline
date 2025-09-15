from importlib import import_module
import inspect

def load_attention(attention_name: str, hidden_dim: int):
    """
    Dynamically load an attention class from models/attention folder.
    """
    att_module = import_module(f"models.attention.{attention_name}")

    # find first class defined in this module
    for _, cls in inspect.getmembers(att_module, inspect.isclass):
        if cls.__module__ == att_module.__name__:
            return cls(hidden_dim)

    raise ValueError(f"No attention class found in {attention_name}")