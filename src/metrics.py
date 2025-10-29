import numpy as np
from sklearn.metrics import roc_auc_score

def auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

def lift_at_k(y_true, y_score, k=0.1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    cut = max(1, int(np.ceil(k * n)))
    idx = np.argsort(-y_score)[:cut]
    top_pos_rate = y_true[idx].mean()
    base_rate = y_true.mean()
    return (top_pos_rate / base_rate) if base_rate > 0 else np.nan

def precision_at_k(y_true, y_score, k=0.1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    cut = max(1, int(np.ceil(k * n)))
    idx = np.argsort(-y_score)[:cut]
    return y_true[idx].mean()