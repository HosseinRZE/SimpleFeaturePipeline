from typing import List, Dict, Any, Tuple
import numpy as np

def create_sequences_by_time(state: Dict[str, Any]) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[int], List[int]]:
    """
    Core sequencer function. Iterates through labels and creates corresponding feature sequences.
    """
    df_data = state['df_data']
    df_labels = state['df_labels']
    lineprice_cols = state.get('lineprice_cols', [c for c in df_labels.columns if c.startswith("linePrice")])

    X_list, y_list, x_lengths, y_lengths = [], [], [], []

    for _, row in df_labels.iterrows():
        # Create a mask to select the time window for the current label
        mask = (df_data["timestamp"] >= row["startTime"]) & (df_data["timestamp"] <= row["endTime"])
        df_sequence = df_data.loc[mask].copy()

        if df_sequence.empty:
            continue
            
        # Store features as a dictionary of numpy arrays
        feature_cols = [c for c in df_sequence.columns if c != "timestamp"]
        X_dict = {'main': df_sequence[feature_cols].values.astype(np.float32)}
        
        # Process labels
        line_prices = row[lineprice_cols].fillna(0).values.astype(np.float32)
        
        X_list.append(X_dict)
        y_list.append(line_prices)
        x_lengths.append(len(df_sequence))
        y_lengths.append((line_prices != 0).sum())

    state['feature_cols'] = feature_cols
    return X_list, y_list, x_lengths, y_lengths