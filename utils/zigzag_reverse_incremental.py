import pandas as pd

def calc_dev(base_price: float, price: float) -> float:
    return 100 * ((price - base_price) / abs(base_price))


class Pivot:
    def __init__(self, price, index, is_high):
        self.price = price
        self.index = index
        self.is_high = is_high

    def is_more_price(self, point):
        return self.price < point if self.is_high else self.price > point


class ZigZag:
    def __init__(self, window_size=3, dev_threshold=2, shadow_mode=True, column="close"):
        self.window_size = window_size
        self.dev_threshold = dev_threshold
        self.shadow_mode = shadow_mode
        self.column = column

        self.price_data = None
        self.zigzag_list = []
        self.last_pivot = None

    def reset(self, initial_data):
        """Initialize with the reversed price data (latest first)."""
        self.price_data = initial_data.copy()
        self.zigzag_list = []
        self.last_pivot = None

    def _check_extremum(self, df_window):
        """Check if the middle candle of the window is a peak or valley."""
        center_idx = df_window.index[len(df_window) // 2]

        if self.shadow_mode:
            # Shadow mode: use highs for peaks, lows for valleys
            center_high = df_window.loc[center_idx, "high"]
            center_low = df_window.loc[center_idx, "low"]

            if center_high == df_window["high"].max():
                return Pivot(center_high, center_idx, True)
            if center_low == df_window["low"].min():
                return Pivot(center_low, center_idx, False)

        else:
            # Close-only mode: use self.column for both peaks & valleys
            center_val = df_window.loc[center_idx, self.column]

            if center_val == df_window[self.column].max():
                return Pivot(center_val, center_idx, True)
            if center_val == df_window[self.column].min():
                return Pivot(center_val, center_idx, False)

        return None

    def update_with_new_candle(self, candle):
        """Add new candle (backward stepping) and update zigzag incrementally."""
        self.price_data = pd.concat(
            [self.price_data, pd.DataFrame([candle])],
            ignore_index=True
        )

        if len(self.price_data) >= self.window_size:
            recent_window = self.price_data.tail(self.window_size)
            new_pivot = self._check_extremum(recent_window)

            if new_pivot:
                if self.last_pivot is None:
                    self._new_pivot_found(new_pivot)
                elif self.last_pivot.is_high == new_pivot.is_high:
                    # Same type — replace if better
                    if self.last_pivot.is_more_price(new_pivot.price):
                        self._update_last_pivot(new_pivot)
                else:
                    # Different type — check deviation
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
                "time": int(self.price_data.iloc[p.index]["timestamp"]),
                "value": p.price,
            }
            for p in self.zigzag_list
        ]

