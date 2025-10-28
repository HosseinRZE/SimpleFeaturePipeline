Here's a comprehensive README.md file for your library, explaining its architecture, components, and usage.

-----

# Feature Pipeline & Add-On Framework

This library provides a flexible, pluggable architecture for building sequential data processing pipelines. The core idea is to separate the pipeline's execution "stages" from the "logic" that runs at each stage. This is achieved through a `FeaturePipeline` orchestrator and a system of `AddOn` plugins.

## 🎯 Core Concepts

1.  **`FeaturePipeline`**: This is the main orchestrator. It defines a series of *stages* (e.g., `run_transformation`, `run_on_evaluation`). Each stage is a method on the `FeaturePipeline` class.
2.  **`BaseAddOn`**: This is an interface (a base class) that defines all the available *hooks* (e.g., `transformation`, `on_evaluation`).
3.  **Custom Add-Ons**: You create your own logic by inheriting from `BaseAddOn` and overriding the hook methods you care about. For example, `MyDataCleanerAddOn` might only implement the `transformation` hook.
4.  **`@run` Decorator**: This is the magic that connects everything. It decorates the pipeline's *stage* methods and tells them which *hook* to run.
      * When you call `pipeline.run_transformation(...)`, the `@run` decorator (with `hook_name="transformation"`) automatically finds *all* registered add-ons that have implemented the `transformation` hook and executes them in order.

This design allows you to build a complex pipeline by simply writing small, focused `AddOn` classes and plugging them into the `FeaturePipeline`.

## ✨ Key Features

  * **Pluggable Architecture**: Easily add, remove, or swap functionality by changing the list of add-ons passed to the pipeline.
  * **Declarative Stages**: Pipeline stages are clearly defined methods on the `FeaturePipeline`, linked to hooks via the `@run` decorator.
  * **Multiple Execution Modes**:
      * **`mode="loop"`**: (Default) Runs the hook for *all* add-ons that implement it. Ideal for transformations or logging.
      * **`mode="priority_loop"`**: Runs the hook for all implementing add-ons, but sorts them based on a `[hook_name]_priority` attribute first. Perfect for evaluation steps where order matters.
      * **`mode="one_time"`**: Enforces that *only one* add-on can implement this hook. Throws an error if zero or more than one are found. Ideal for steps like data loading or final output formatting.
  * **Execution Tracing**: A powerful `@trace` decorator can wrap any function that uses a pipeline. It automatically logs all *tracked* (`track=True`) steps, including which add-on ran, its priority (if any), and how long it took.
  * **Pipeline Inspection**: The `pipeline.method_table()` utility prints a clear table showing every pipeline stage, its corresponding hook, and which of your *currently loaded* add-ons implement it.

-----

## 🚀 Usage Example

Let's walk through a complete example of setting up and running a pipeline.

### Step 1: Define Your Custom Add-Ons

First, create classes that inherit from `BaseAddOn` and implement the logic you need.

```python
# (Assuming 'BaseAddOn' is available from 'add_ons.base_addon')
from add_ons.base_addon import BaseAddOn
from typing import Dict, Any

class DataLoadingAddOn(BaseAddOn):
    """
    Implements the 'sequence_on_train' hook, which is a 'one_time' hook.
    This will be the only add-on allowed to implement this.
    """
    def sequence_on_train(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        print("DataLoadingAddOn: Loading initial data...")
        state['data'] = [1, 2, 3, 4, 5]
        
        # Add a message to the trace log
        pipeline_extra_info['current_trace_message'] = "Loaded 5 records"
        return state

class DataCleaningAddOn(BaseAddOn):
    """
    Implements the 'transformation' hook, a 'loop' hook.
    It will run alongside any other add-on that also implements 'transformation'.
    """
    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        print("DataCleaningAddOn: Cleaning data...")
        state['data'] = [x for x in state.get('data', []) if x > 2]
        pipeline_extra_info['current_trace_message'] = f"Removed {5 - len(state['data'])} items"
        return state

class FeatureEngineeringAddOn(BaseAddOn):
    """
    Also implements the 'transformation' hook. It will run in sequence
    with DataCleaningAddOn.
    """
    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        print("FeatureEngineeringAddOn: Generating features...")
        state['features_squared'] = [x*x for x in state.get('data', [])]
        pipeline_extra_info['current_trace_message'] = "Created 'features_squared'"
        return state

class HighPriorityEvaluator(BaseAddOn):
    """
    Implements a 'priority_loop' hook. The priority attribute
    name MUST match the hook name.
    """
    on_evaluation_priority = 100 # Runs first

    def on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        print("HighPriorityEvaluator: Calculating main score...")
        eval_data['score'] = 0.99
        pipeline_extra_info['current_trace_message'] = "Score=0.99"
        return eval_data

class LowPriorityLogger(BaseAddOn):
    """
    Also implements 'on_evaluation', but with a lower priority.
    It will run *after* HighPriorityEvaluator.
    """
    on_evaluation_priority = 10 # Runs later

    def on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        print(f"LowPriorityLogger: Logging score: {eval_data.get('score')}")
        pipeline_extra_info['current_trace_message'] = "Logged score to console"
        return eval_data
```

### Step 2: Build and Inspect the Pipeline

Now, instantiate your add-ons and pass them to the `FeaturePipeline`.

```python
# (Assuming 'FeaturePipeline' is available)
from your_library.feature_pipeline import FeaturePipeline

# 1. Initialize all your add-ons
add_ons = [
    DataLoadingAddOn(),
    DataCleaningAddOn(),
    FeatureEngineeringAddOn(),
    HighPriorityEvaluator(),
    LowPriorityLogger()
]

# 2. Create the pipeline instance
pipeline = FeaturePipeline(add_ons=add_ons)

# 3. Inspect the pipeline to see who implements what
print("--- Inspecting Pipeline Setup ---")
pipeline.method_table(show=True)
```

**Output of `method_table()`:**

```
=========================================================================================================================================
--- Add-On Hook Implementation for Pipeline ---
╒═════════╤═══════════════════════╤═══════════════════════╤═══════════════════╤═════════════════════════════════════════════════════════════╕
│   Order │ Pipeline Method       │ Hook Name             │ Mode              │ Implementing Add-Ons                                        │
╞═════════╪═══════════════════════╪═══════════════════════╪═══════════════════╪═════════════════════════════════════════════════════════════╡
│      10 │ run_before_sequence   │ before_sequence       │ loop              │ ---                                                         │
├─────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────────────────────────────────────────────────┤
│      20 │ run_sequence_on_train │ sequence_on_train     │ one_time          │ DataLoadingAddOn                                            │
├─────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────────────────────────────────────────────────┤
│ ...     │ ...                   │ ...                   │ ...               │ ...                                                         │
├─────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────────────────────────────────────────────────┤
│      50 │ run_transformation    │ transformation        │ loop              │ DataCleaningAddOn, FeatureEngineeringAddOn                  │
├─────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────────────────────────────────────────────────┤
│      60 │ run_on_evaluation     │ on_evaluation         │ priority_loop     │ HighPriorityEvaluator, LowPriorityLogger                    │
├─────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────────────────────────────────────────────────┤
│ ...     │ ...                   │ ...                   │ ...               │ ...                                                         │
╘═════════╧═══════════════════════╧═══════════════════════╧═══════════════════╧═════════════════════════════════════════════════════════════╛
=========================================================================================================================================
```

### Step 3: Run the Pipeline with Tracing

Finally, use the `@trace` decorator on a function that calls your pipeline stages.

```python
# (Assuming '@trace' is available from 'utils.decorators.trace')
from your_library.utils.decorators.trace import trace

@trace(time_track=True, log_level="INFO")
def run_my_training_job(pipeline: FeaturePipeline):
    """
    A function that runs the pipeline stages, wrapped by @trace.
    """
    print("\n--- STARTING TRACED JOB ---")
    
    # This dictionary is passed to all add-ons
    pipeline_info = {} 
    
    # 1. Start with an empty state
    state = {}
    
    # 2. Run the 'one_time' data loader
    # This will be tracked because @run(track=True)
    state = pipeline.run_sequence_on_train(state, pipeline_info)
    
    # 3. Run the 'loop' transformations
    # This will log *both* add-ons
    state = pipeline.run_transformation(state, pipeline_info)
    
    # 4. Run the 'priority_loop' evaluations
    # The trace will show them running in priority order
    eval_state = pipeline.run_on_evaluation(state, pipeline_info)
    
    print("--- TRACED JOB COMPLETE ---")
    print(f"Final State: {eval_state}")


# --- EXECUTE ---
run_my_training_job(pipeline)
```

**Output of `run_my_training_job()`:**

```
--- Inspecting Pipeline Setup ---
... (method table) ...

--- STARTING TRACED JOB ---
DataLoadingAddOn: Loading initial data...
DataCleaningAddOn: Cleaning data...
FeatureEngineeringAddOn: Generating features...
HighPriorityEvaluator: Calculating main score...
LowPriorityLogger: Logging score: 0.99
--- TRACED JOB COMPLETE ---
Final State: {'data': [3, 4, 5], 'features_squared': [9, 16, 25], 'score': 0.99}

===================================================================================================================================================================
--- Trace Log for: run_my_training_job ---
╒═══════════════════════╤═══════════════════════════╤══════════════════════════════════╤═════════════════════╕
│ Method                │ Add-On                    │ Message                          │   Time Elapsed (s) │
╞═══════════════════════╪═══════════════════════════╪══════════════════════════════════╪═════════════════════╡
│ run_sequence_on_train │ DataLoadingAddOn          │ Loaded 5 records                 │             0.0001 │
├───────────────────────┼───────────────────────────┼──────────────────────────────────┼────────────────────┤
│ run_transformation    │ DataCleaningAddOn         │ Removed 2 items                  │             0.0000 │
├───────────────────────┼───────────────────────────┼──────────────────────────────────┼────────────────────┤
│ run_transformation    │ FeatureEngineeringAddOn   │ Created 'features_squared'       │             0.0000 │
├───────────────────────┼───────────────────────────┼──────────────────────────────────┼────────────────────┤
│ run_on_evaluation     │ HighPriorityEvaluator     │ Score=0.99                       │             0.0001 │
├───────────────────────┼───────────────────────────┼──────────────────────────────────┼────────────────────┤
│ run_on_evaluation     │ LowPriorityLogger         │ Logged score to console          │             0.0000 │
╘═══════════════════════╧═══════════════════════════╧══════════════════════════════════╧═════════════════════╛
===================================================================================================================================================================
```

-----

## API Reference

### `FeaturePipeline`

The main orchestrator class.

  * `__init__(self, add_ons: List[BaseAddOn] = None)`: Initializes the pipeline with a list of add-on instances.
  * `method_table(self, show: bool = True) -> str`: Generates and optionally prints a table of all pipeline stages and their implementing add-ons.
  * `run_... (self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]`: A series of pipeline stage methods (e.g., `run_before_sequence`, `run_transformation`, `run_on_server_request`). These are the methods you call to execute a stage.

### `BaseAddOn`

The base class for all plugins.

  * Inherit from this class to create your own add-on.
  * Override its methods (hooks) to inject logic into the pipeline.
  * **Available Hooks**:
      * `before_sequence`
      * `apply_window`
      * `transformation`
      * `on_evaluation`
      * `on_evaluation_end`
      * `sequence_on_train`
      * `sequence_on_server`
      * `on_server_init`
      * `on_first_request`
      * `on_server_request`
      * `on_server_inference`
      * `on_final_output`

### Decorators

#### `@run(hook_name: str, mode: str, track: bool, order: int)`

*Used inside the `FeaturePipeline` class to define a stage.*

  * `hook_name (str)`: The name of the `BaseAddOn` method to execute (e.g., `"transformation"`).
  * `mode (str)`: The execution strategy. Must be one of `"loop"`, `"priority_loop"`, or `"one_time"`.
  * `track (bool)`: If `True`, this stage will be logged by the `@trace` decorator.
  * `order (int)`: A number used to sort the stages for display in `method_table`.

#### `@trace(time_track: bool = False, log_level: str = None)`

*Used to wrap a function that *calls* pipeline methods.*

  * `time_track (bool)`: If `True`, the final trace table will include an execution time for each step.
  * `log_level (str)`: If set (e.g., `"INFO"`, `"DEBUG"`), the trace table will also be written to the `logging` system at that level.