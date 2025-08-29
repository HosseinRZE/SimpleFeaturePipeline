import pandas as pd
import talib

class CandleLabeler:
    def __init__(self):
        # Define patterns with TA-Lib function and human-readable name
        # Single letters reused from original for known patterns
        self.patterns = {
            "H": (talib.CDLHAMMER, "Hammer"),
            "h": (talib.CDLHANGINGMAN, "Hanging Man"),
            "E": (talib.CDLENGULFING, "Engulfing Pattern"),
            "S": (talib.CDLSHOOTINGSTAR, "Shooting Star"),
            "M": (talib.CDLMORNINGSTAR, "Morning Star"),
            "N": (talib.CDLEVENINGSTAR, "Evening Star"),
            "D": (talib.CDLDOJI, "Doji"),
            # Additional patterns
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

    def get_legend(self):
        return {k: v[1] for k, v in self.patterns.items()}

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # initialize empty list for each row
        labels = pd.Series([[] for _ in range(len(df))], index=df.index)

        for letter, (func, _) in self.patterns.items():
            out = func(df.open, df.high, df.low, df.close)
            # append letter to list where pattern matched
            for i, val in enumerate(out):
                if val != 0:
                    labels[i].append(letter)

        df["label"] = labels
        return df
