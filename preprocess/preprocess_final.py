from typing import List, Dict, Any, Tuple
import pandas as pd
from utils.debug_samples2 import _debug_sample_check
from add_ons.feature_pipeline_base import FeaturePipeline
from utils.decorators.trace import trace
# Import the custom data structures
from data_structure.sequence_sample import SequenceSample
from data_structure.sequence_collection import SequenceCollection # CRITICAL NEW IMPORT

@trace(time_track=True)
       # --- 3. Main Pipeline Orchestrator ---
def preprocess_pipeline(
    data_csv: str,
    labels_csv: str,
    feature_pipeline: FeaturePipeline,
    val_split: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    for_torch: bool = True, # True for PyTorch Dataset structure, False for NumPy arrays
    debug_indices: List[int] = None
) -> Tuple:
    """
    Executes a modular, multi-step preprocessing pipeline. 
    
    This orchestrator loads raw data, coordinates the execution of all AddOns
    defined in the FeaturePipeline, and ensures all feature data, labels, and 
    denormalization factors remain perfectly aligned using the SequenceCollection 
    object.

    Args:
        data_csv (str): Path to the CSV file containing the time series feature data.
        labels_csv (str): Path to the CSV file containing the labeled events and time windows.
        feature_pipeline (FeaturePipeline): Object containing all feature engineering AddOns.
        val_split (bool, optional): If True, splits data into train/validation/test sets. 
                                    Defaults to True.
        test_size (float, optional): Proportion of data to reserve for the test set (and val if val_split is True). 
                                     Defaults to 0.2.
        random_state (int, optional): Seed for reproducibility of the data split. Defaults to 42.
        for_torch (bool, optional): If True, prepares data for PyTorch Dataset creation. 
                                    If False, returns raw NumPy arrays. Defaults to True.
        debug_indices (List[int], optional): A list of original row indices to debug and check 
                                             the state of the corresponding SequenceSample. Defaults to None.

    Returns:
        Tuple: A tuple containing the processed datasets/arrays and associated metadata.
    
    Raises:
        TypeError: If the sequencer method fails to return a SequenceCollection object, 
                   breaking the pipeline's core data contract.
    """
    # --- Step 0: Load Initial Data ---
    df_data = pd.read_csv(data_csv)
    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"])

    df_labels = pd.read_csv(labels_csv)
    df_labels["startTime"] = pd.to_datetime(df_labels["startTime"], unit="s")
    df_labels["endTime"] = pd.to_datetime(df_labels["endTime"], unit="s")
    
    # --- Step 1: Initiate State ---
    # The 'state' dictionary is passed and modified throughout the pipeline.
    state: Dict[str, Any] = {
        'df_data': df_data,
        'df_labels': df_labels,
        # 'samples' is RESERVED for the SequenceCollection object
        'samples': None 
    }
    
    # --- Step 2: Global Pre-processing Hooks (e.g., target factor calculation) ---
    state = feature_pipeline.run_before_sequence(state, feature_pipeline.extra_info)
    
    # --- Step 3: Sequence Generation ---
    # The sequencer must iterate over df_labels, create SequenceSample objects, 
    # wrap them in a SequenceCollection, and store it in state['samples'].
    state = feature_pipeline.run_sequence_on_train(state, feature_pipeline.extra_info)
    
    # --- Step 3.1: CONTRACT VALIDATION (MANDATORY CHECK) ---
    # Ensures the sequencer followed the required data contract.
    if not isinstance(state.get('samples'), SequenceCollection):
        raise TypeError(
            "The 'sequencer' method must return a 'state' dictionary containing "
            "the key 'samples' which holds a 'SequenceCollection' object. "
            f"Found type: {type(state.get('samples'))}"
        )

    print(f"✅ Sequencer returned {len(state['samples'])} aligned sequences in SequenceCollection.")
    
    # --- Step 4: Run Hooks on Feature Windows (Modifies X and metadata of each SequenceSample) ---
    state = feature_pipeline.run_apply_window(state, feature_pipeline.extra_info)
    
    # --- Step 5: Filter Invalid Sequences (Removes invalid SequenceSample objects) ---
    # The filter utility must use the SequenceCollection.filter() method to rebuild 
    # the collection, which automatically updates the internal index map.
    # state = filter_invalid_sequences(state, feature_pipeline.extra_info) 
    

    # --- Step 6: Final Transformation (e.g., NumPy conversion, padding) ---
    state = feature_pipeline.run_transformation(state, feature_pipeline.extra_info)

    # --- Step 7: Debug Sample Check ---
    if debug_indices:
        _debug_sample_check(debug_indices, state)

    # --- Step 8: Output ---
    # _prepare_output must now extract the X, y, and index data from the SequenceCollection 
    # before splitting and packaging.
    state = feature_pipeline.run_final_output(
        state,
        feature_pipeline.extra_info,
        val_split=val_split,
        test_size=test_size,
        random_state=random_state,
        for_torch=for_torch
    )
    return state["final_output"]