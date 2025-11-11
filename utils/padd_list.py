import numpy as np

def padding_list(list_of_arrays, pad_value=0.0):
    max_len = max(len(x) for x in list_of_arrays)
    padded = np.array([
        np.pad(x, (0, max_len - len(x)), constant_values=pad_value)
        for x in list_of_arrays
    ], dtype=np.float32)
    return padded, max_len