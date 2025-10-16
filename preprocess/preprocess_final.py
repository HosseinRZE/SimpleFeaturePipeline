# --- 3. Main Pipeline Orchestrator ---

def preprocess_pipeline(
    data_csv: str,
    labels_csv: str,
    feature_pipeline: FeaturePipeline,
    val_split: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    for_torch: bool = True, # Use True for build_multiinput_dataset, False for flat NumPy array
    debug_indices: List[int] = None
) -> Tuple:
    """
    Executes a modular, multi-step preprocessing pipeline.

    - Steps are defined by 'add-on' modules within the FeaturePipeline.
    - The process flows from initial data loading to final formatted output.
    """
    # --- Step 0: Load Initial Data ---
    df_data = pd.read_csv(data_csv)
    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"])

    df_labels = pd.read_csv(labels_csv)
    df_labels["startTime"] = pd.to_datetime(df_labels["startTime"], unit="s")
    df_labels["endTime"] = pd.to_datetime(df_labels["endTime"], unit="s")
    
    # --- Step 1: Initiate State ---
    # The 'state' dictionary is passed and modified through the entire pipeline.
    state = {
        'df_data': df_data,
        'df_labels': df_labels,
        'lineprice_cols': [c for c in df_labels.columns if c.startswith("linePrice")],
        **feature_pipeline.extra_params
    }
    
    # --- Step 2: Before Sequence ---
    # Apply add-ons that work on the full, un-sequenced DataFrames.
    for addon in feature_pipeline.add_ons:
        state = addon.before_sequence(state)
        
    # --- Step 3: Sequencer Function ---
    # Create lists of sequences based on label time windows.
    X_list, y_list, x_lengths, y_lengths = _create_sequences(state)
    state.update({
        'X_list': X_list,
        'y_list': y_list,
        'x_lengths': x_lengths,
        'y_lengths': y_lengths,
    })

    # --- Step 4: Apply Window ---
    # Apply add-ons that process each sequence individually.
    for addon in feature_pipeline.add_ons:
        state = addon.apply_window(state)

    # --- Step 5: Transformation ---
    # Apply add-ons that might change the shape/structure of the data.
    for addon in feature_pipeline.add_ons:
        state = addon.transformation(state)

    # --- Step 6: Debug Sample Check ---
    if debug_indices:
        _debug_sample_check(debug_indices, state)

    # --- Step 7: Output ---
    # Format the final data into datasets or arrays and handle splitting.
    return _prepare_output(state, val_split, test_size, random_state, for_torch)