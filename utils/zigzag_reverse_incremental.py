import pandas as pd
from datetime import datetime

def calc_dev(base_price: float, price: float) -> float:
    return 100 * ((price - base_price) / abs(base_price))

class Pivot:
    def __init__(self, price, index, is_high):
        self.price = price
        self.index = index
        self.is_high = is_high

    def is_more_price(self, point):
        """Check if new point is 'better' than current pivot."""
        return self.price < point if self.is_high else self.price > point

class ZigZag:
    def __init__(self, window_size=3, dev_threshold=1, shadow_mode=True,
                 column="close", max_pivots=None, max_steps=None):
        self.window_size = window_size
        self.dev_threshold = dev_threshold
        self.shadow_mode = shadow_mode
        self.column = column
        self.max_pivots = max_pivots
        self.max_steps = max_steps

        self.price_data = None
        self.zigzag_list = []
        self.last_pivot = None
        self.steps_taken = 0

    def reset(self, initial_data):
        """
        Initialize with the reversed price data (latest first).
        initial_data: DataFrame with timestamp, open, high, low, close, volume
        """
        df = initial_data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        self.price_data = df
        self.zigzag_list = []
        self.last_pivot = None
        self.steps_taken = 0

    def _check_extremum(self, df_window):
        """Check if the middle candle of the window is a peak or valley."""
        center_idx = df_window.index[len(df_window) // 2]

        if self.shadow_mode:
            # Use highs for peaks, lows for valleys
            center_high = df_window.loc[center_idx, "high"]
            center_low = df_window.loc[center_idx, "low"]

            if center_high == df_window["high"].max():
                return Pivot(center_high, center_idx, True)
            if center_low == df_window["low"].min():
                return Pivot(center_low, center_idx, False)

        else:
            # Use close prices only
            center_val = df_window.loc[center_idx, self.column]

            if center_val == df_window[self.column].max():
                return Pivot(center_val, center_idx, True)
            if center_val == df_window[self.column].min():
                return Pivot(center_val, center_idx, False)

        return None

    def update_with_new_candle(self, candle):
        """
        Add new candle (backward stepping) and update zigzag incrementally.
        Candle: dict with keys matching dataset format.
        """
        if self.max_steps is not None and self.steps_taken >= self.max_steps:
            return  # Stop after reaching step limit

        # Convert timestamp to datetime
        candle["timestamp"] = pd.to_datetime(candle["timestamp"])

        # Append at the end because we already reversed data initially
        self.price_data = pd.concat(
            [self.price_data, pd.DataFrame([candle])],
            ignore_index=True
        )

        self.steps_taken += 1

        # Stop if pivot limit reached
        if self.max_pivots is not None and len(self.zigzag_list) >= self.max_pivots:
            return

        # Only check last `window_size` candles
        if len(self.price_data) >= self.window_size:
            recent_window = self.price_data.tail(self.window_size)
            new_pivot = self._check_extremum(recent_window)

            if new_pivot:
                if self.last_pivot is None:
                    self._new_pivot_found(new_pivot)
                elif self.last_pivot.is_high == new_pivot.is_high:
                    # Same type (peak-peak or valley-valley)
                    if self.last_pivot.is_more_price(new_pivot.price):
                        self._update_last_pivot(new_pivot)
                else:
                    # Different type, check deviation
                    dev = abs(calc_dev(self.last_pivot.price, new_pivot.price))
                    if dev >= self.dev_threshold:
                        self._new_pivot_found(new_pivot)

    def _new_pivot_found(self, pivot):
        self.zigzag_list.append(pivot)
        self.last_pivot = pivot

    def _update_last_pivot(self, pivot):
        if self.zigzag_list:
            self.zigzag_list[-1] = pivot
        self.last_pivot = pivot

    def get_pivots(self):
        """Return pivots as list of dicts for chart drawing."""
        return [
            {
                "time": self.price_data.iloc[p.index]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "value": p.price,
            }
            for p in self.zigzag_list
        ]
