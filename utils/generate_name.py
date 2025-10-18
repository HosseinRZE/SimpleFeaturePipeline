import inspect
from datetime import datetime

def generate_filenames(name_ext_pairs):
    """
    Generate timestamped filenames and a folder name based on the calling function.

    Args:
        name_ext_pairs (list): A list where each element is either:
            - (name, ext): for files to generate
            - or a single string: used as folder name (last element)

    Returns:
        tuple: filenames in order + folder_name at the end
    """
    # Identify the calling function (one frame back)
    caller_name = inspect.stack()[1].function

    # Timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If the last item is a string → folder name base
    if isinstance(name_ext_pairs[-1], str):
        folder_base = name_ext_pairs[-1]
        name_ext_pairs = name_ext_pairs[:-1]
    else:
        folder_base = caller_name  # fallback if not provided

    # Create folder name
    folder_name = f"{folder_base}_{caller_name}_{timestamp}"

    # Generate filenames for the rest
    filenames = []
    for name, ext in name_ext_pairs:
        ext = ext.lstrip('.')  # normalize extension
        filenames.append(f"{name}_{caller_name}_{timestamp}.{ext}")

    return (*filenames, folder_name)
