from typing import Dict, Any
import numpy as np
import pandas as pd

class SequenceSample:
    """
    An atomic, self-contained unit of data representing a single time series sequence 
    or event.

    The primary role of this class is to permanently bundle all components 
    (features, labels, and critical metadata) for one sample, guaranteeing 
    data alignment throughout the entire preprocessing and filtering pipeline.
    """
    def __init__(self, 
                 original_index: int, 
                 X_features: Dict[str, np.ndarray], 
                 y_labels: np.ndarray, 
                 metadata: Dict[str, Any]):
        """
        Initializes a SequenceSample.

        Args:
            original_index (int): The index of the corresponding row in the 
                                  original df_labels DataFrame. This serves as 
                                  the permanent, unchangeable primary key for 
                                  lookups and alignment with global data (e.g., 
                                  the SequenceCollection's index map).
            X_features (Dict[str, np.ndarray]): The input features for the model. 
                                                It is a dictionary mapping a 
                                                feature group name (e.g., 'main', 
                                                'aux') to its time-series NumPy 
                                                array (T x F).
            y_labels (np.ndarray): The target label(s) for the sequence, typically 
                                   a 1D NumPy array.
            metadata (Dict[str, Any]): A dictionary for storing sequence-specific 
                                       data, such as denormalization factors 
                                       (e.g., last close price), local scaling 
                                       statistics (e.g., min/max for this window), 
                                       or timestamps. Data placed here remains 
                                       aligned with the sample at all times.
        """
        self.original_index = original_index
        self.X: Dict[str, np.ndarray] = X_features 
        self.y: np.ndarray = y_labels
        self.metadata: Dict[str, Any] = metadata

    def __repr__(self) -> str:
        """Provides a concise, informative representation of the sample."""
        x_shapes = {k: v.shape for k, v in self.X.items()}
        return (f"<SequenceSample idx={self.original_index} | X={x_shapes} | Y={self.y.shape}>")

    def get_denorm_factor(self, key: str) -> float:
        """
        Convenience method to retrieve a denormalization factor from metadata.

        Args:
            key (str): The key under which the factor is stored.

        Returns:
            float: The stored factor, or 1.0 if the key is not found (default no-op factor).
        """
        return self.metadata.get(key, 1.0)
    
    @property
    def x_lengths(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping each feature group key → number of time steps (rows).

        Returns
        -------
        Dict[str, int]
            Example:
            {
                "main": 128,
                "aux": 128
            }
        """
        lengths = {}
        for group_key, data in self.X.items():
            if isinstance(data, (pd.DataFrame, pd.Series)):
                lengths[group_key] = len(data)
            elif isinstance(data, np.ndarray):
                lengths[group_key] = data.shape[0]
            else:
                lengths[group_key] = 0
        return lengths