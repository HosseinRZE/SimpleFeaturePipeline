from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd 

def create_sequences_by_time(state: Dict[str, Any]) -> Tuple[List[Dict[str, pd.DataFrame]], List[np.ndarray]]:
    """
    Core sequencer function. Iterates through labels and creates corresponding feature sequences.
    """
    df_data = state['df_data']
    df_labels = state['df_labels']
    lineprice_cols = state.get('lineprice_cols', [c for c in df_labels.columns if c.startswith("linePrice")])

    X_list, y_list, x_lengths, y_lengths = [], [], [], []
    feature_cols = []

    for _, row in df_labels.iterrows():
        mask = (df_data["timestamp"] >= row["startTime"]) & (df_data["timestamp"] <= row["endTime"])
        df_sequence = df_data.loc[mask].copy()

        if df_sequence.empty:
            continue
        
        if not feature_cols: 
            feature_cols = [c for c in df_sequence.columns if c != "timestamp"]
        
        X_dict = {'main': df_sequence[feature_cols]}
        
        # 🟢 THE FIX: Explicitly cast to float before filling NaNs.
        # This resolves the ambiguity that causes the FutureWarning.
        line_prices = row[lineprice_cols].astype(np.float32).fillna(0).values
        
        X_list.append(X_dict)
        y_list.append(line_prices)
        x_lengths.append(len(df_sequence))
        y_lengths.append((line_prices != 0).sum())

    state['feature_cols'] = feature_cols
    state['x_lengths'] = x_lengths
    state['y_lengths'] = y_lengths

    return X_list, y_list