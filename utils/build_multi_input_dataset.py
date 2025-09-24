from utils.multi_input_dataset import MultiInputDataset

def build_multiinput_dataset(X_dicts_list, y_arr, x_lengths=None):
    """
    Convert a list of X_dicts and label array into a MultiInputDataset.

    Parameters
    ----------
    X_dicts_list : list of dict
        Each element is a dict of arrays per input channel.
    y_arr : np.ndarray
        Label array of shape (n_samples, n_classes)
    x_lengths : list or np.ndarray, optional
        Sequence lengths for each sample (used for padding-aware models)

    Returns
    -------
    MultiInputDataset
    """
    if x_lengths is None:
        x_lengths = [len(X_dicts_list[i]["main"]) for i in range(len(X_dicts_list))]

    X_dict = {k: [d[k] for d in X_dicts_list] for k in X_dicts_list[0]}
    return MultiInputDataset(X_dict, y_arr, x_lengths)
