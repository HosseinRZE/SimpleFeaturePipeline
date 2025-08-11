import pandas as pd
import plotly.graph_objects as go

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
    def __init__(self, data, window_size=7, dev_threshold=3,
                 first_skip=False, last_skip=True, shadow_mode=True,
                 column="close", list_return=False):
        """
        data: Pandas DataFrame (must contain 'open','high','low','close')
        window_size: rolling window size for local extrema
        dev_threshold: minimum % change between pivot points
        first_skip: allow incomplete first window
        last_skip: allow incomplete last window
        shadow_mode: calculate from high/low instead of 'close'
        column: column name for non-shadow mode
        list_return: return pivot info as list of lists
        """
        self.price_data = data
        self.window_size = window_size
        self.dev_threshold = dev_threshold
        self.first_skip = first_skip
        self.last_skip = last_skip
        self.shadow_mode = shadow_mode
        self.column = column
        self.list_return = list_return
        self.reset_internal_state()

    def reset_internal_state(self):
        self.zigzag_list = []
        self.peaks_and_valleys_list = []
        self.peaks = []
        self.valleys = []

    def _find_peaks_valleys(self, window, mode):
        """Store local peaks or valleys if they match window center."""
        center_idx = len(window) // 2
        center_val = window.iloc[center_idx]

        if mode == "peaks" and center_val == window.max():
            self.peaks.append([center_val, window.idxmax(), True])
        elif mode == "valleys" and center_val == window.min():
            self.valleys.append([center_val, window.idxmin(), False])

        return 1  # Required for rolling.apply()

    def pivot_finder(self):
        self.first_index = self.price_data.index[0]
        self.last_index = self.price_data.index[-1]

        if self.shadow_mode:
            self.price_data["high"].rolling(self.window_size).apply(
                self._find_peaks_valleys, args=("peaks",))
            self.price_data["low"].rolling(self.window_size).apply(
                self._find_peaks_valleys, args=("valleys",))
            combined = self.peaks + self.valleys
            self.peaks_and_valleys_list = sorted(combined, key=lambda x: x[1])
        else:
            self.price_data[self.column].rolling(self.window_size).apply(
                self._find_peaks_valleys, args=("mixed",))

        return [Pivot(price, idx, is_high) for price, idx, is_high in self.peaks_and_valleys_list]

    def zigzag_pivots(self, pivots):
        self.last_pivot = None
        self.last_low_pivot = None
        self.last_high_pivot = None

        for pivot in pivots:
            if self.last_pivot is None:
                if pivot.is_high:
                    self.last_high_pivot = pivot
                else:
                    self.last_low_pivot = pivot

                # Check if deviation threshold met
                for lp in [self.last_low_pivot, self.last_high_pivot]:
                    if lp and lp.is_high != pivot.is_high:
                        dev = abs(calc_dev(lp.price, pivot.price))
                        if dev >= self.dev_threshold:
                            self.new_pivot_found(lp.index, lp.price, lp.is_high)
                            self.new_pivot_found(pivot.index, pivot.price, pivot.is_high)
            else:
                if self.last_pivot.is_high == pivot.is_high:
                    if self.last_pivot.is_more_price(pivot.price):
                        self.update_last_pivot(pivot)
                else:
                    dev = abs(calc_dev(self.last_pivot.price, pivot.price))
                    if dev >= self.dev_threshold:
                        self.new_pivot_found(pivot.index, pivot.price, pivot.is_high)

        return self.zigzag_list

    def update_last_pivot(self, pivot):
        pv = Pivot(pivot.price, pivot.index, pivot.is_high)
        if self.zigzag_list:
            self.zigzag_list[-1] = pv
        self.last_pivot = pv

    def new_pivot_found(self, index, price, is_high):
        p = Pivot(price, index, is_high)
        self.zigzag_list.append(p)
        self.last_pivot = p

    def give_zigzag(self):
        self.reset_internal_state()
        pivots = self.pivot_finder()
        zz_list = self.zigzag_pivots(pivots)
        if self.list_return:
            return [[p.price, p.index, p.is_high] for p in zz_list]
        return zz_list

    def plot_ohlc_zigzag(self, zigzag_list):
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=self.price_data.index,
            open=self.price_data['open'],
            high=self.price_data['high'],
            low=self.price_data['low'],
            close=self.price_data['close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='OHLC'
        ))

        zz_x = [p.index for p in zigzag_list]
        zz_y = [p.price for p in zigzag_list]
        fig.add_trace(go.Scatter(
            x=zz_x,
            y=zz_y,
            mode='lines+markers',
            line=dict(color='red', dash='dash'),
            name='Zigzag'
        ))

        fig.update_layout(
            title='OHLC Chart with Zigzag',
            xaxis_title='Index',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )

        # Save HTML file instead of showing directly
        fig.write_html("zigzag_plot.html")
        print("Plot saved as zigzag_plot.html — open it in a browser.")



# Example incremental usage
if __name__ == "__main__":
    price_data = pd.read_csv("/home/iatell/financial_data/BTC15min.csv")
    price_data.rename(columns={
        'column6': 'volume',
        'column5': 'close',
        'column4': 'low',
        'column3': 'high',
        'column2': 'open'
    }, inplace=True)

    start, step = 0, 300
    while True:
        subset = price_data.iloc[start:start+step]
        zz = ZigZag(subset, window_size=3, dev_threshold=3)
        zz_list = zz.give_zigzag()
        zz.plot_ohlc_zigzag(zz_list)
        cont = input("Press 'n' for next batch, else quit: ")
        if cont.lower() != 'n':
            break
        start += step
