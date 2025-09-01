import numpy as np
from sklearn.metrics import f1_score

def tune_thresholds(y_true, y_probs, step=0.05):
    """
    Tune thresholds per label to maximize F1-score.

    Args:
        y_true: array-like, shape (n_samples, n_labels)
        y_probs: array-like, shape (n_samples, n_labels) probabilities
        step: granularity for threshold search (default 0.05)

    Returns:
        thresholds: list of optimal thresholds per label
    """
    n_labels = y_true.shape[1]
    thresholds = []

    for i in range(n_labels):
        best_thresh = 0.5
        best_f1 = 0.0
        for t in np.arange(0.1, 0.9 + step, step):
            preds = (y_probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        thresholds.append(best_thresh)
    return thresholds

def tune_thresholds_nn(y_true, y_probs, step=0.05):
    """
    Tune optimal thresholds per label to maximize F1-score.
    """
    n_labels = y_true.shape[1]
    thresholds = []

    for i in range(n_labels):
        best_thresh = 0.5
        best_f1 = 0.0
        for t in np.arange(0.1, 0.9 + step, step):
            preds = (y_probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        thresholds.append(best_thresh)
    return thresholds