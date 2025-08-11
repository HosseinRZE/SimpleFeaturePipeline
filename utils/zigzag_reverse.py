from flask import Flask, render_template, jsonify
import pandas as pd
import time

# --- ZigZag Classes ---
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
                 shadow_mode=True, column="close", list_return=False):
        self.price_data = data
        self.window_size = window_size
        self.dev_threshold = dev_threshold
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
        center_idx = len(window) // 2
        center_val = window.iloc[center_idx]

        if mode == "peaks" and center_val == window.max():
            self.peaks.append([center_val, window.idxmax(), True])
        elif mode == "valleys" and center_val == window.min():
            self.valleys.append([center_val, window.idxmin(), False])
        return 1

    def pivot_finder(self):
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

    def zigzag_pivots_backward(self, pivots):
        self.last_pivot = None
        self.zigzag_list = []
     
        for pivot in (pivots):
 
            if self.last_pivot is None:
                self.new_pivot_found(pivot.index, pivot.price, pivot.is_high)
            else:
                if self.last_pivot.is_high == pivot.is_high:
                    # Same type, replace if better
                    if self.last_pivot.is_more_price(pivot.price):
                        self.update_last_pivot(pivot)
               
                else:
                    dev = abs(calc_dev(self.last_pivot.price, pivot.price))
                    if dev >= self.dev_threshold:
                        if pivot.is_high != self.last_pivot.is_high:
                            self.new_pivot_found(pivot.index, pivot.price, pivot.is_high)


        self.zigzag_list.reverse()
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

    def give_zigzag(self, backward=False):
        self.reset_internal_state()
        pivots = self.pivot_finder()
        if backward:
            zz_list = self.zigzag_pivots_backward(pivots)
        else:
            raise NotImplementedError("Forward mode not used in this example")
        if self.list_return:
            return [[p.price, p.index, p.is_high] for p in zz_list]
        return zz_list


