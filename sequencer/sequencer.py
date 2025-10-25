from typing import List, Dict, Any
import numpy as np
import pandas as pd
# Assuming these are available in the running environment
from data_structure.sequence_sample import SequenceSample 
from data_structure.sequence_collection import SequenceCollection 

def create_sequences_by_time(
    state: Dict[str, Any], 
    extra_info: Dict[str, Any], 
    include_cols: List[str] = None, 
    exclude_cols: List[str] = None
) -> Dict[str, Any]:
    """
    Core sequencer function responsible for generating aligned sequence samples.

    It extracts the corresponding feature window from the raw data (df_data), 
    filters columns based on explicit 'include' and 'exclude' lists, and creates 
    a single SequenceSample object for each valid sequence.

    Args:
        state (Dict[str, Any]): The current pipeline state containing 'df_data' 
                                and 'df_labels'.
        extra_info (Dict[str, Any]): The persistent feature pipeline configuration/factors.
        include_cols (List[str], optional): Explicit list of columns to INCLUDE. If provided, 
                                            only these columns (that also survive exclusion) 
                                            will be present. Defaults to None (no inclusion filter).
        exclude_cols (List[str], optional): List of columns to EXCLUDE. Defaults to None.

    Returns:
        Dict[str, Any]: The updated state dictionary containing the mandatory 
                        'samples' key populated with a SequenceCollection object.
    """
    df_data: pd.DataFrame = state['df_data']
    df_labels: pd.DataFrame = state['df_labels']
    
    # Initialize parameters
    include_cols = include_cols or []
    exclude_cols = exclude_cols or []
    
    # Identify the relevant columns for labels (Y)
    lineprice_cols = state.get('lineprice_cols', [c for c in df_labels.columns if c.startswith("linePrice")])

    # The list that will hold all the atomic SequenceSample objects
    samples: List[SequenceSample] = []
    
    # feature_cols will hold the final, canonical list of columns for the features
    feature_cols = []

    # Iterate through df_labels, capturing the original index (row.name)
    for original_index, row in df_labels.iterrows():
        # 1. Define the Feature Window (X)
        mask = (df_data["timestamp"] >= row["startTime"]) & (df_data["timestamp"] <= row["endTime"])
        df_sequence = df_data.loc[mask].copy()

        if df_sequence.empty:
            continue
        
        # Determine feature columns only once
        if not feature_cols: 
            all_potential_cols = [c for c in df_sequence.columns]
            
            # 1. Apply exclusions (always exclude 'timestamp')
            cols_after_exclude = [c for c in all_potential_cols if c not in exclude_cols and c != "timestamp"]
            
            # 2. Apply inclusions
            if include_cols:
                # Intersection: Only keep columns that are both filtered and explicitly included
                feature_cols = [c for c in cols_after_exclude if c in include_cols]
            else:
                feature_cols = cols_after_exclude
        
        # --- Feature Extraction ---
        # Extract the window as a DataFrame using the determined feature_cols list
        X_df = df_sequence[feature_cols].astype(np.float32)
        X_dict = {'main': X_df} 
        
        # 2. Extract Label Data (Y)
        y_labels = row[lineprice_cols].astype(np.float32).fillna(0).values
        
        # 3. Create the SequenceSample Object
        sample = SequenceSample(
            original_index=original_index,
            X_features=X_dict, 
            y_labels=y_labels,
            metadata={} 
        )
        
        # 4. Collect the fully aligned object
        samples.append(sample)

    # --- Final State Update ---
    state['samples'] = SequenceCollection(samples) 
    
    return state
