from typing import List, Dict, Any, Tuple, Callable
import pandas as pd
import numpy as np

class DataStoreMock:
    """
    A simple class to hold the historical data (candles) received so far, 
    mimicking a database or real-time data accumulator.
    """
    def __init__(self, initial_df: pd.DataFrame = None):
        self.dataset = initial_df.iloc[0:0].copy() if initial_df is not None else pd.DataFrame()
        if not self.dataset.index.name:
            self.dataset.index.name = 'timestamp'
            
    def add_candle(self, candle_row: pd.Series):
        """Adds a new candle (row) to the dataset."""
        self.dataset.loc[candle_row.name] = candle_row
        self.dataset = self.dataset.sort_index()

    @property
    def current_data(self) -> pd.DataFrame:
        """Returns the current complete dataset."""
        return self.dataset

    def get_last_n(self, n: int) -> pd.DataFrame:
        """Returns the last 'n' rows of the dataset."""
        return self.dataset.iloc[-n:]
    
    def __len__(self):
        return len(self.dataset)

# Helper function (to replace ServerPreprocess.prepare_seq)
# This logic will need to be absorbed by an AddOn running in 'on_server_request'
def get_sequence_for_model(df_data: pd.DataFrame, seq_len: int, features: List[str]) -> Dict[str, pd.DataFrame]:
    """
    This is a simplified *example* of what an AddOn must now do. 
    It will be replaced by the AddOn logic in a real application.
    """
    if len(df_data) < seq_len:
        raise ValueError(f"Not enough data ({len(df_data)} records) to form a sequence of length {seq_len}.")
    
    # Simple example: just return the features for the last seq_len rows
    sequence_data = df_data[features].iloc[-seq_len:]
    
    # Return the dictionary format expected by the model's forward method
    # This is a placeholder structure; actual implementation depends on your model
    return {"main": sequence_data}