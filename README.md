````markdown
# User Guide: The Feature Pipeline & Add-On System

This guide explains how to use the `FeaturePipeline` system to create and debug custom data processing steps.

The system is built on two core concepts:
1.  **`FeaturePipeline`**: The main "engine" that runs a series of processing steps in a fixed order.
2.  **`BaseAddOn`**: "Plugins" you create to hook into those processing steps and perform your custom logic.



[Image of a data pipeline diagram]


## Core Concepts

### The `FeaturePipeline`
The `FeaturePipeline` class is the orchestra conductor. It doesn't perform any data processing itself; it just calls a series of "hooks" (methods) in a predefined order.

When you call a method like `pipeline.run_transformation(...)`, the pipeline finds all registered Add-Ons that have implemented the `transformation` hook and runs them.

### The `BaseAddOn`
The `BaseAddOn` is the "plugin" class you will use to build all of your custom logic.

Think of it as a template with many empty methods. You create your own class that inherits from `BaseAddOn` and just "fill in" the methods you care about.

For example:
* To create a new feature, you would implement the `transformation` hook.
* To filter out bad data, you might also implement the `transformation` hook or the earlier `apply_window` hook.
* If you don't implement a hook (like `on_evaluation`), your Add-On is simply skipped during that step.

---

## 🛠️ How to Create a Custom Add-On

Creating a new processing step is simple.

### Step 1: Create Your Add-On Class
Create a new Python file (e.g., `my_addons.py`) and define your class inheriting from `BaseAddOn`.

Let's create an Add-On that drops a list of columns.

```python
from add_ons.base_addon import BaseAddOn
from data_structure.sequence_collection import SequenceCollection
import pandas as pd
from typing import Dict, Any, List

class ColumnDropperAddOn(BaseAddOn):
    def __init__(self, columns_to_drop: List[str]):
        # You can add your own __init__
        self.columns_to_drop = columns_to_drop

    def transformation(self, 
                         state: Dict[str, Any], 
                         pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        This hook is called by the pipeline's 'run_transformation' step.
        """
        
        # 1. Get the data from the state
        samples: SequenceCollection = state.get('samples')
        if not samples:
            return state # Always return the state

        # 2. Perform your logic
        for sample in samples:
            # Loop through each feature group (e.g., 'main', 'aux')
            for group_key, data in sample.X.items():
                if isinstance(data, pd.DataFrame):
                    # Drop the columns if they exist
                    cols_to_drop = [col for col in self.columns_to_drop if col in data.columns]
                    if cols_to_drop:
                        sample.X[group_key] = data.drop(columns=cols_to_drop)

        # 3. Always return the modified state
        return state
````

### Step 2: Register Your Add-On in the Pipeline

In your main script, import your new Add-On and pass an instance of it to the `FeaturePipeline` constructor.

```python
from feature_pipeline import FeaturePipeline
from my_addons import ColumnDropperAddOn
from root_power_mapper_addon import RootPowerMapperAddOn # Your other add-on

# 1. Create instances of all the add-ons you want to use
addon_1 = ColumnDropperAddOn(columns_to_drop=['volume', 'trade_count'])
addon_2 = RootPowerMapperAddOn(p=0.25, main=['open_prop', 'close_prop'])

# 2. Pass the list of add-ons to the pipeline
my_pipeline = FeaturePipeline(add_ons=[
    addon_1,
    addon_2
])

# 3. Run the pipeline (example for training)
# (Assuming 'initial_state' is a dict with your raw data)
my_pipeline.preprocess_pipeline(initial_state, my_pipeline.extra_info)
```

The pipeline will now automatically run your `ColumnDropperAddOn` during the `run_transformation` step.

-----

## 🐞 Debugging with the `@trace` Decorator

The trace system is your primary tool for debugging your Add-Ons. It generates a summary table of every step that ran, which Add-On ran, how long it took, and any debug messages you logged.

### 1\. Activating the Trace

To turn on the trace, you **must** add the `@trace()` decorator to the main pipeline method you are calling (e.g., `preprocess_pipeline`).

```python
# In your FeaturePipeline class file...
from utils.decorators.trace import trace
from utils.decorators.run import run

class FeaturePipeline:
    def __init__(self, add_ons: List[BaseAddOn] = None):
        # ...
    
    # ... (all your @run methods) ...

    # This is your main "runner" method
    # Add the @trace decorator here!
    @trace(time_track=True)
    def preprocess_pipeline(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]):
        """
        This is an example of a main pipeline runner.
        """
        state = self.run_before_sequence(state, pipeline_extra_info)
        state = self.run_sequence_on_train(state, pipeline_extra_info)
        state = self.run_apply_window(state, pipeline_extra_info)
        state = self.run_transformation(state, pipeline_extra_info)
        state = self.run_final_output(state, pipeline_extra_info)
        return state
```

  * `@trace()`: Turns on the trace table.
  * `@trace(time_track=True)`: Also records the time each step took.

### 2\. Logging from Your Add-On

Inside your Add-On, use the `self.add_trace_print()` method to log a message to the trace table.

This is the **only way** to get debug messages into the final report.

```python
# Inside your ColumnDropperAddOn.transformation method...

    def transformation(self, 
                         state: Dict[str, Any], 
                         pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        
        samples: SequenceCollection = state.get('samples')
        if not samples:
            # You can log messages even when returning early
            self.add_trace_print(pipeline_extra_info, "Skipped: No samples found in state.")
            return state

        for sample in samples:
            for group_key, data in sample.X.items():
                if isinstance(data, pd.DataFrame):
                    cols_to_drop = [col for col in self.columns_to_drop if col in data.columns]
                    
                    if cols_to_drop:
                        # ✨ THIS IS THE LOGGING CALL ✨
                        self.add_trace_print(
                            pipeline_extra_info, 
                            f"Group '{group_key}': Dropping {len(cols_to_drop)} cols."
                        )
                        sample.X[group_key] = data.drop(columns=cols_to_drop)

        return state
```

**Note:** You *must* pass `pipeline_extra_info` to `add_trace_print`. This is the "bridge" that allows the Add-On to send the message back to the decorator.

### 3\. Reading the Output

When you run your pipeline (e.g., `my_pipeline.preprocess_pipeline(...)`), the trace table will be printed at the very end.

**Example Output:**

```
--- Trace Log for: preprocess_pipeline ---
╒═══════════════════════╤═════════════════════════════╤════════════════════════════════════════════════════════════════════════╤════════════════════╕
│ Method                │ Add-On                      │ Message                                                                │   Time Elapsed (s) │
╞═══════════════════════╪═════════════════════════════╪════════════════════════════════════════════════════════════════════════╪════════════════════╡
│ run_sequence_on_train │ SequencerAddOn              │                                                                        │             0.1304 │
├───────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────┤
│ run_transformation    │ ColumnDropperAddOn          │ Group 'main': Dropping 2 cols.                                         │             0.0210 │
├───────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────┤
│ run_transformation    │ RootPowerMapperAddOn        │ First value trace: 'close_prop': x=1.087713 -> f(x)=0.544210 (p=0.250, │             0.0367 │
│                       │                             │ M=1.0000)                                                              │                    │
├───────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────┤
│ run_final_output      │ PrepareOutputAddOn          │                                                                        │             0.0010 │
╘═══════════════════════╧═════════════════════════════╧════════════════════════════════════════════════════════════════════════╧════════════════════╛
```

  * **Method:** The `FeaturePipeline` method that was run.
  * **Add-On:** The name of your class that was executed.
  * **Message:** The string you provided to `self.add_trace_print()`.

> **Pro Tip: Message Overwriting**
>
> The trace system shows **only one message per Add-On per step**. If you call `self.add_trace_print()` twice inside the *same* `transformation` method, the **second message will overwrite the first one**.
>
> To log multiple things, combine them into one string:
>
> ```python
> msg1 = "Found 100 samples."
> msg2 = "Normalization complete."
> self.add_trace_print(pipeline_extra_info, f"{msg1}\n{msg2}")
> ```

-----

## 📚 Reference

### Pipeline Hooks (from `FeaturePipeline`)

| Pipeline Method (`@run`) | Hook Name (in `BaseAddOn`) | Mode | Description |
| :--- | :--- | :--- | :--- |
| `run_before_sequence` | `before_sequence` | `loop` | Runs before any sequencing. |
| `run_sequence_on_train` | `sequence_on_train` | `one_time` | **(Special)** Only one Add-On can implement this. |
| `run_sequence_on_server` | `sequence_on_server` | `loop` | Runs during server sequencing. |
| `run_apply_window` | `apply_window` | `loop` | Runs after sequencing, on each sample. |
| `run_transformation` | `transformation` | `loop` | Main step for feature engineering. |
| `run_on_evaluation` | `on_evaluation` | `priority_loop` | Runs during evaluation. (Requires `self.on_evaluation_priority`) |
| `run_on_evaluation_end` | `on_evaluation_end` | `priority_loop` | Runs at end of evaluation. (Requires `self.on_evaluation_end_priority`) |
| `run_on_server_init` | `on_server_init` | `loop` | Runs once when the inference server starts. |
| `run_on_first_request` | `on_first_request` | `loop` | Runs on the first server request. |
| `run_on_server_request` | `on_server_request` | `loop` | Runs on every server request. |
| `run_on_server_inference` | `on_server_inference` | `loop` | Runs after the model predicts. |
| `run_final_output` | `on_final_output` | `one_time` | **(Special)** Final step before returning data. |

### `BaseAddOn` Public Methods

  * `add_trace_print(pipeline_extra_info: Dict, message: str)`:
      * Logs a `message` string to the trace table for the current step.
      * `pipeline_extra_info` is always required.

<!-- end list -->

```
```