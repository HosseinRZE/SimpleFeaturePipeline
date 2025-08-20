import numpy as np
import pandas as pd
from collections import deque

class Pivot:
    def __init__(self, price, index, is_high: bool):
        self.price = price
        self.index = index
        self.is_high = is_high

    def is_more_price(self, point: float) -> bool:
        """Check if a new candidate price should replace this pivot"""
        return self.price < point if self.is_high else self.price > point


class ZigZag:
    def __init__(self, window_size=3, dev_threshold=1.0, column="close",
                 max_pivots=10, stationary=False,
                 include_last_candle_as_pivot=True,
                 include_distances=True,
                 shadow_mode=False):
        self.window_size = window_size
        self.dev_threshold = dev_threshold
        self.column = column
        self.max_pivots = max_pivots
        self.stationary = stationary
        self.include_last_candle_as_pivot = include_last_candle_as_pivot
        self.include_distances = include_distances
        self.shadow_mode = shadow_mode

        self.buffer = deque(maxlen=window_size)  # each entry: (idx, high, low, close)
        self.pivots: list[Pivot] = []

    def _detect_local_extremum(self):
        """Check center candle in buffer for local high/low."""
        if len(self.buffer) < self.window_size:
            return None

        mid_idx = self.window_size // 2
        mid_candle = self.buffer[mid_idx]  # (idx, high, low, close)
        idx, high, low, close = mid_candle

        if self.shadow_mode:
            # Compare highs for peaks
            left_highs = [c[1] for c in list(self.buffer)[:mid_idx]]
            right_highs = [c[1] for c in list(self.buffer)[mid_idx+1:]]
            is_high = all(high > h for h in left_highs + right_highs)

            # Compare lows for valleys
            left_lows = [c[2] for c in list(self.buffer)[:mid_idx]]
            right_lows = [c[2] for c in list(self.buffer)[mid_idx+1:]]
            is_low = all(low < l for l in left_lows + right_lows)

            if is_high:
                return Pivot(high, idx, is_high=True)
            if is_low:
                return Pivot(low, idx, is_high=False)
        else:
            # Use close for both high/low detection
            left = [c[3] for c in list(self.buffer)[:mid_idx]]
            right = [c[3] for c in list(self.buffer)[mid_idx+1:]]
            is_high = all(close > p for p in left + right)
            is_low = all(close < p for p in left + right)

            if is_high or is_low:
                return Pivot(close, idx, is_high=is_high)

        return None

    def _add_or_replace_pivot(self, new_pivot: Pivot):
        if not self.pivots:
            self.pivots.append(new_pivot)
            return

        last = self.pivots[-1]

        # Same type -> maybe replace
        if last.is_high == new_pivot.is_high:
            if last.is_more_price(new_pivot.price):
                self.pivots[-1] = new_pivot
            return

        # Different type -> check threshold
        if abs(new_pivot.price - last.price) >= self.dev_threshold:
            self.pivots.append(new_pivot)

    def update(self, idx, high, low, close):
        """Process one new candle with OHLC"""
        self.buffer.append((idx, high, low, close))

        pivot = self._detect_local_extremum()
        if pivot:
            self._add_or_replace_pivot(pivot)

    def get_features(self, current_price, current_index):
        """Return pivots as dict of zigzag_x and zigzag_dist_x"""
        pivots = self.pivots.copy()

        # Optionally include last candle
        if self.include_last_candle_as_pivot:
            pivots = pivots + [Pivot(current_price, current_index, False)]

        # Pad pivots to max_pivots
        if len(pivots) < self.max_pivots:
            pad = [Pivot(current_price, -1, False)] * (self.max_pivots - len(pivots))
            pivots = pad + pivots

        pivots = pivots[-self.max_pivots:]

        # Stationary normalization
        if self.stationary:
            pivots = [Pivot(p.price / current_price, p.index, p.is_high) for p in pivots]

        # Build feature dict
        features = {}
        for i, p in enumerate(reversed(pivots), 1):
            features[f"zigzag_{i}"] = p.price
            if self.include_distances:
                features[f"zigzag_dist_{i}"] = abs(p.index - current_index)

        return features

    def draw(self, df, max_index=None):
        """
        Return a list of dicts for plotting: [{'time': timestamp, 'value': price}, ...]
        df: DataFrame with at least a 'timestamp' column
        max_index: optional max pivot index to include
        """
        draw_pivots = []
        for p in self.pivots[-self.max_pivots:]:
            if max_index is None or 0 <= p.index <= max_index:
                draw_pivots.append({
                    "time": int(pd.Timestamp(df.loc[p.index, "timestamp"]).timestamp()),
                    "value": float(p.price)
                })
        return draw_pivots
