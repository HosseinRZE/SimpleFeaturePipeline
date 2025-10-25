# utils/decorators/trace.py

import logging
import time
from functools import wraps
from typing import Callable, Any
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _find_pipeline_instance(args: tuple, kwargs: dict) -> Any:
    """Helper to find the FeaturePipeline instance in function arguments."""
    is_pipeline = lambda obj: hasattr(obj, 'add_ons') and hasattr(obj, 'sequencer_fn')
    for arg in args:
        if is_pipeline(arg):
            return arg
    for val in kwargs.values():
        if is_pipeline(val):
            return val
    return None

def trace(time_track: bool = False, log_level: str = None):
    """
    Manages and displays a trace log of pipeline steps.

    This decorator sets up the logging "context" that the @run decorator
    will write to.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pipeline_instance = _find_pipeline_instance(args, kwargs)

            if not pipeline_instance:
                return func(*args, **kwargs)

            # 1. SETUP: Attach log and config to the pipeline instance
            # The @run decorator will look for these attributes.
            pipeline_instance._trace_log = []
            pipeline_instance._trace_config = {
                'time_track': time_track,
                'log_level': log_level.upper() if log_level else None
            }

            try:
                # 2. EXECUTION: Run the decorated function (e.g., preprocess_pipeline)
                return func(*args, **kwargs)
            finally:
                # 3. TEARDOWN: Format and display the results
                log_data = pipeline_instance._trace_log
                config = pipeline_instance._trace_config
                header_text = f"--- Trace Log for: {func.__name__} ---"
                
                print("\n" + "="*len(header_text))
                print(header_text)

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

                    table = tabulate(table_data, headers=headers, tablefmt="fancy_grid")
                    print(table)

                    if config['log_level']:
                        log_method = getattr(logger, config['log_level'].lower(), None)
                        if log_method:
                            log_method(f"Trace log for {func.__name__}:\n{table}")

                print("="*len(header_text) + "\n")

                # 4. CLEANUP: Remove temporary attributes
                delattr(pipeline_instance, '_trace_log')
                delattr(pipeline_instance, '_trace_config')
        return wrapper
    return decorator