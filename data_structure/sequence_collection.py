from typing import List, Dict, Any, Optional
from data_structure.sequence_sample import SequenceSample
import pandas as pd
import numpy as np

class SequenceCollection:
    """
    A collection class to hold all SequenceSample objects and provide a fast, 
    indexed lookup mechanism by the original index (O(1) access).

    This object is the central data container for the entire dataset pipeline.
    It now includes Pandas-like inspection methods for ease of use.
    """
    def __init__(self, samples: List[SequenceSample]):
        self._samples = samples
        self._index_map = self._build_index_map(samples)

    def _build_index_map(self, samples: List[SequenceSample]) -> Dict[int, SequenceSample]:
        """Creates the internal O(1) lookup map from the original_index."""
        return {sample.original_index: sample for sample in samples}

    def get_by_original_index(self, original_index: int) -> Optional[SequenceSample]:
        """
        Retrieves a SequenceSample directly using its original row index.
        This enables clean debugging lookup: state['samples'].get_by_original_index(12).
        """
        return self._index_map.get(original_index)

    def get_list(self) -> List[SequenceSample]:
        """Returns the internal list for ordered iteration and traditional processing."""
        return self._samples

    def __len__(self) -> int:
        """Allows len(collection)"""
        return len(self._samples)

    def __iter__(self):
        """Allows direct iteration over the samples list."""
        return iter(self._samples)

    def filter(self, samples_to_keep: List[SequenceSample]) -> 'SequenceCollection':
        """
        Rebuilds the collection with a filtered list of samples. 
        This is how the filter_invalid_sequences utility will update the collection.
        
        Note: This creates a new SequenceCollection instance, ensuring immutability 
        of the collection object when filtering.
        """
        return SequenceCollection(samples_to_keep)

    def __repr__(self) -> str:
        return f"<SequenceCollection | Total Samples: {len(self)}>"

    def __getitem__(self, idx: int) -> SequenceSample:
        """Enable list-style indexing, e.g., samples[0]."""
        return self._samples[idx]

    # --------------------------------------------------------------------------
    # NEW METHODS ADDED BELOW
    # --------------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        """
        Returns the number of samples in the collection, mimicking the (N, ) shape 
        of a dataset.

        Returns:
            tuple: A tuple (N, ) where N is the total number of samples.
        """
        return (len(self),)
    
    def drop(self, indices_to_drop: List[int]) -> 'SequenceCollection':
        """
        Creates a new SequenceCollection by dropping samples based on their 
        **original_index** values.

        This provides a convenient, readable way to remove specific samples.

        Args:
            indices_to_drop (List[int]): A list of `original_index` values 
                                         to exclude from the new collection.

        Returns:
            SequenceCollection: A **new** collection instance containing all samples 
                                whose `original_index` is NOT in the list.
        """
        # Create a set for O(1) average time lookup
        drop_set = set(indices_to_drop)
        
        # Filter the internal list
        samples_to_keep = [
            sample for sample in self._samples 
            if sample.original_index not in drop_set
        ]
        
        # Use the existing filter method to build a new collection instance
        return self.filter(samples_to_keep)

    def head(self, n: int = 5) -> List[SequenceSample]:
        """
        Returns the first N samples in the ordered collection list, similar to 
        `pandas.DataFrame.head()`.

        Args:
            n (int, optional): The number of samples to return from the beginning. 
                               Defaults to 5.

        Returns:
            List[SequenceSample]: A list of the first N samples.
        """
        return self._samples[:n]

    def tail(self, n: int = 5) -> List[SequenceSample]:
        """
        Returns the last N samples in the ordered collection list, similar to 
        `pandas.DataFrame.tail()`.

        Args:
            n (int, optional): The number of samples to return from the end. 
                               Defaults to 5.

        Returns:
            List[SequenceSample]: A list of the last N samples.
        """
        return self._samples[-n:]