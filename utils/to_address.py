from io import StringIO
import pandas as pd
def to_address(df: pd.DataFrame):
    """
    Convert a DataFrame into an in-memory CSV buffer 
    that can be passed to functions expecting a CSV path.

    Args:
        df (pd.DataFrame): The labels DataFrame.

    Returns:
        StringIO: File-like object containing CSV data.
    """
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)  # reset pointer so read_csv starts at beginning
    return buffer
