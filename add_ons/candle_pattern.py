from typing import Dict, Any,List, Set
from add_ons.base_addon import BaseAddOn
import pandas as pd
import numpy as np
from data_structure.sequence_collection import SequenceCollection
import talib

class CandleLabeler:
    def __init__(self):
        # Define patterns with TA-Lib function and human-readable name
        self.patterns = {
            "H": (talib.CDLHAMMER, "Hammer"),
            "h": (talib.CDLHANGINGMAN, "Hanging Man"),
            "E": (talib.CDLENGULFING, "Engulfing Pattern"),
            "S": (talib.CDLSHOOTINGSTAR, "Shooting Star"),
            "M": (talib.CDLMORNINGSTAR, "Morning Star"),
            "N": (talib.CDLEVENINGSTAR, "Evening Star"),
            "D": (talib.CDLDOJI, "Doji"),
            "2": (talib.CDL2CROWS, "Two Crows"),
            "3": (talib.CDL3BLACKCROWS, "Three Black Crows"),
            "I": (talib.CDL3INSIDE, "Three Inside Up/Down"),
            "O": (talib.CDL3LINESTRIKE, "Three Outside Up/Down"),
            "T": (talib.CDL3STARSINSOUTH, "Three Stars In The South"),
            "W": (talib.CDL3WHITESOLDIERS, "Three Advancing White Soldiers"),
            "A": (talib.CDLABANDONEDBABY, "Abandoned Baby"),
            "B": (talib.CDLADVANCEBLOCK, "Advance Block"),
            "L": (talib.CDLBELTHOLD, "Belt-hold"),
            "K": (talib.CDLBREAKAWAY, "Breakaway"),
            "C": (talib.CDLCLOSINGMARUBOZU, "Closing Marubozu"),
            "V": (talib.CDLCONCEALBABYSWALL, "Concealing Baby Swallow"),
            "Q": (talib.CDLCOUNTERATTACK, "Counterattack"),
            "F": (talib.CDLDARKCLOUDCOVER, "Dark Cloud Cover"),
            "*": (talib.CDLDOJISTAR, "Doji Star"),
            "G": (talib.CDLDRAGONFLYDOJI, "Dragonfly Doji"),
            "J": (talib.CDLEVENINGDOJISTAR, "Evening Doji Star"),
            "Y": (talib.CDLGAPSIDESIDEWHITE, "Up/Down-gap side-by-side white lines"),
            "Z": (talib.CDLGRAVESTONEDOJI, "Gravestone Doji"),
            "R": (talib.CDLHARAMI, "Harami Pattern"),
            "X": (talib.CDLHARAMICROSS, "Harami Cross Pattern"),
            "+": (talib.CDLHIGHWAVE, "High-Wave Candle"),
            "^": (talib.CDLHIKKAKE, "Hikkake Pattern"),
            "&": (talib.CDLHIKKAKEMOD, "Modified Hikkake Pattern"),
            "$": (talib.CDLHOMINGPIGEON, "Homing Pigeon"),
            "%": (talib.CDLIDENTICAL3CROWS, "Identical Three Crows"),
            "=": (talib.CDLINNECK, "In-Neck Pattern"),
            "~": (talib.CDLINVERTEDHAMMER, "Inverted Hammer"),
            "!": (talib.CDLKICKING, "Kicking"),
            "@": (talib.CDLKICKINGBYLENGTH, "Kicking - bull/bear determined by longer marubozu"),
            "#": (talib.CDLLADDERBOTTOM, "Ladder Bottom"),
            "?": (talib.CDLLONGLEGGEDDOJI, "Long Legged Doji"),
            "/": (talib.CDLLONGLINE, "Long Line Candle"),
            "m": (talib.CDLMARUBOZU, "Marubozu"),
            "q": (talib.CDLMATCHINGLOW, "Matching Low"),
            "w": (talib.CDLMATHOLD, "Mat Hold"),
            "o": (talib.CDLMORNINGDOJISTAR, "Morning Doji Star"),
            "p": (talib.CDLONNECK, "On-Neck Pattern"),
            "r": (talib.CDLPIERCING, "Piercing Pattern"),
            "s": (talib.CDLRICKSHAWMAN, "Rickshaw Man"),
            "f": (talib.CDLRISEFALL3METHODS, "Rising/Falling Three Methods"),
            "g": (talib.CDLSEPARATINGLINES, "Separating Lines"),
            "c": (talib.CDLSHORTLINE, "Short Line Candle"),
            "v": (talib.CDLSPINNINGTOP, "Spinning Top"),
            "b": (talib.CDLSTALLEDPATTERN, "Stalled Pattern"),
            "n": (talib.CDLSTICKSANDWICH, "Stick Sandwich"),
            "t": (talib.CDLTAKURI, "Takuri (Dragonfly Doji with long lower shadow)"),
            "u": (talib.CDLTASUKIGAP, "Tasuki Gap"),
            "j": (talib.CDLTHRUSTING, "Thrusting Pattern"),
            "e": (talib.CDLTRISTAR, "Tristar Pattern"),
            "i": (talib.CDLUNIQUE3RIVER, "Unique 3 River"),
            "y": (talib.CDLUPSIDEGAP2CROWS, "Upside Gap Two Crows"),
            "z": (talib.CDLXSIDEGAP3METHODS, "Upside/Downside Gap Three Methods"),
        }

    def get_legend(self) -> Dict[str, str]:
        return {k: v[1] for k, v in self.patterns.items()}

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        required_cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required_cols):
            rename_map = {
                "Open": "open", "High": "high", "Low": "low", "Close": "close",
                "Timestamp": "timestamp", "Volume": "volume"
            }
            df = df.rename(columns=rename_map)
            df.columns = [c.lower() for c in df.columns]
            
            if not all(c in df.columns for c in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}. Found: {df.columns.tolist()}")

        # initialize empty list for each row
        labels = pd.Series([[] for _ in range(len(df))], index=df.index)

        for letter, (func, _) in self.patterns.items():
            try:
                out = func(df.open, df.high, df.low, df.close)
                for i, val in enumerate(out):
                    if val != 0:
                        labels.iloc[i].append(letter)
            except Exception as e:
                print(f"Error processing pattern {self.patterns[letter][1]}: {e}")
        
        df["label_list"] = labels # Changed column name to be clear
        return df# Define key for storing the column names
    
COLUMN_MAP_KEY = 'candle_pattern_columns'

class CandlePatternEncoderAddOn(BaseAddOn):
    """
    Finds specific candlestick patterns and encodes them as a multi-hot
    pandas DataFrame. This DataFrame is added to sample.X under a new key.
    """
    def __init__(self, 
                 patterns: List[str], 
                 target_group: str = 'main',
                 new_feature_key: str = 'candle_pattern'):
        """
        Initializes the add-on.
        
        Args:
            patterns: A list of pattern *names* (e.g., "Hammer", "Doji")
                      The order of this list will be preserved.
            target_group: The key in sample.X holding the source OHLCV DataFrame.
            new_feature_key: The new key to add to sample.X (e.g., 'candle_pattern')
                             The value will be a pd.DataFrame of shape (N, len(patterns)).
        """
        super().__init__()
        self.target_group = target_group
        self.new_feature_key = new_feature_key
        
        self.labeler = CandleLabeler()
        full_legend = self.labeler.get_legend()
        
        reverse_legend: Dict[str, str] = {v: k for k, v in full_legend.items()}
        
        self.pattern_names_ordered: List[str] = []
        self.pattern_letters_ordered: List[str] = []
        
        for name in patterns:
            if name in reverse_legend:
                self.pattern_names_ordered.append(name)
                self.pattern_letters_ordered.append(reverse_legend[name])
            else:
                print(f"Warning: Pattern '{name}' not found in CandleLabeler. It will be ignored.")
        
        self.letters_to_check: Set[str] = set(self.pattern_letters_ordered)
        
        print(f"CandlePatternEncoderAddOn initialized. Encoding {len(self.pattern_names_ordered)} patterns.")
        print(f"Order: {self.pattern_names_ordered}")

    def _map_labels_to_vector(self, label_list: List[str]) -> np.ndarray:
        """Helper to map a list of letters to a multi-hot vector."""
        label_set = set(label_list)
        vector = [
            1 if letter in label_set else 0 
            for letter in self.pattern_letters_ordered
        ]
        return np.array(vector, dtype=np.int8)

    def _apply_multi_hot_encoding(self, 
                                  samples: SequenceCollection, 
                                  pipeline_extra_info: Dict[str, Any],
                                  mode: str = "Training") -> bool:
        """
        Core logic for applying the encoding.
        """
        if not samples: 
            self.add_trace_print(pipeline_extra_info, f"Skipped {mode}. No samples found.")
            return False

        self.add_trace_print(pipeline_extra_info, f"Starting multi-hot encoding for {len(samples)} samples.")
        
        for i, sample in enumerate(samples):
            if self.target_group not in sample.X:
                self.add_trace_print(pipeline_extra_info, f"Sample {i}: Skipping, no '{self.target_group}' key.")
                continue
            
            df = sample.X[self.target_group]
            
            # Ensure df is a DataFrame (it could be np.ndarray if already processed)
            if not isinstance(df, pd.DataFrame):
                self.add_trace_print(pipeline_extra_info, f"Sample {i}: Skipping, '{self.target_group}' is not a DataFrame.")
                continue
                
            try:
                # 1. Run CandleLabeler -> adds 'label_list' col
                labeled_df = self.labeler.label_dataframe(df)
                
                # 2. Map list of letters to a multi-hot vector for each row
                
                vector_series = labeled_df['label_list'].apply(self._map_labels_to_vector)
                
                # 3. Stack all row-vectors into a 2D NumPy array
                multi_hot_array = np.stack(vector_series.values)
                
                # 4. *** NEW: Create a DataFrame ***
                pattern_df = pd.DataFrame(
                    multi_hot_array,
                    index=labeled_df.index,  # Use original index (timestamps)
                    columns=self.pattern_names_ordered # Use pattern names
                )
                
                # 5. Add this new DataFrame as a top-level feature
                sample.X[self.new_feature_key] = pattern_df
                
                # 6. Clean up the intermediate 'label_list' column from the main df
                sample.X[self.target_group] = labeled_df.drop(columns=['label_list'])
                
            except Exception as e:
                self.add_trace_print(pipeline_extra_info, f"🔥 {mode} ERROR on sample {i}: {e}. Skipping sample.")
                continue
        
        self.add_trace_print(pipeline_extra_info, f"{mode} complete.")
        return True

    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the multi-hot encoding process during training.
        """
        samples: SequenceCollection = state.get('samples')
        
        self._apply_multi_hot_encoding(samples, pipeline_extra_info, mode="Training")
        
        # We still save this for external components to easily check
        column_key = self.new_feature_key + '_columns'
        pipeline_extra_info[column_key] = self.pattern_names_ordered
        self.add_trace_print(pipeline_extra_info, f"Saved column map to {column_key}")
        
        return state

    def on_server_request(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the multi-hot encoding process during inference.
        """
        samples: SequenceCollection = state.get('samples')
        self._apply_multi_hot_encoding(samples, pipeline_extra_info, mode="Inference")
        return state