import logging
import time
from functools import wraps
from typing import Callable, Any
from tabulate import tabulate
# Removed: tempfile, shutil, pathlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _find_pipeline_instance(args: tuple, kwargs: dict) -> Any:
    """Helper to find the FeaturePipeline instance in function arguments."""
    is_pipeline = lambda obj: hasattr(obj, 'add_ons') and hasattr(obj, 'run_before_sequence')
    
    for arg in args:
        if is_pipeline(arg):
            return arg
    for val in kwargs.values():
        if is_pipeline(val):
            return val
    return None

# --- Reverted to original signature (no 'preserve') ---
def trace(time_track: bool = False, log_level: str = None):
    """
    Manages and displays a trace log of pipeline steps.
    
    Args:
        time_track (bool): Whether to track execution time.
        log_level (str): Logging level.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pipeline_instance = _find_pipeline_instance(args, kwargs)

            if not pipeline_instance:
                return func(*args, **kwargs)

            # --- Reverted to simple config (no 'log_dir' or 'preserve') ---
            pipeline_instance._trace_log = []
            pipeline_instance._trace_config = {
                'time_track': time_track,
                'log_level': log_level.upper() if log_level else None
            }
            # --- End Reversion ---

            try:
                # 2. EXECUTION
                return func(*args, **kwargs)
            finally:
                # 3. TEARDOWN
                log_data = pipeline_instance._trace_log
                config = pipeline_instance._trace_config
                header_text = f"--- Trace Log for: {func.__name__} ---"
                
                print("\n" + "="*len(header_text))
                print(header_text)

                # --- Removed "Artifacts preserved at..." print block ---

                if not log_data:
                    print("No tracked steps were executed.")
                else:
                    headers = ["Method", "Add-On", "Message"]
                    if config['time_track']:
                        headers.append("Time Elapsed (s)")

                    table_data = []
                    for row in log_data:
                        record = [row['method'], row['addon'], row['message']]
                        if config['time_track']:
                            record.append(f"{row.get('time', 0):.4f}")
                        table_data.append(record)

                    # --- Removed 'maxcolwidths' from tabulate ---
                    table = tabulate(
                        table_data, 
                        headers=headers, 
                        tablefmt="fancy_grid"
                    )
                    # --- End Reversion ---
                    
                    print(table)

                    if config['log_level']:
                        log_method = getattr(logger, config['log_level'].lower(), None)
                        if log_method:
                            log_method(f"Trace log for {func.__name__}:\n{table}")

                print("="*len(header_text) + "\n")

                # --- Removed 'shutil.rmtree' cleanup logic ---

                # 4. CLEANUP
                delattr(pipeline_instance, '_trace_log')
                delattr(pipeline_instance, '_trace_config')
                
        return wrapper
    return decorator