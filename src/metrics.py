import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_metric(y_true, y_score, metric_fn, n_bootstrap=1000, 
                     confidence=0.95, seed=42, ci_mode=False):
    """Bootstrap CI for a metric. Uses percentile method."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    
    point = float(metric_fn(y_true, y_score))
    
    if ci_mode:
        n_bootstrap = min(n_bootstrap, 100)
    
    rng = np.random.RandomState(seed)
    vals = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        
        # need both classes for AUC
        if len(np.unique(yt)) < 2:
            continue
        try:
            vals.append(metric_fn(yt, ys))
        except:
            continue
    
    if len(vals) < 10:
        return {"value": round(point, 4), "ci_lower": round(point, 4), "ci_upper": round(point, 4)}
    
    alpha = 1 - confidence
    lo = np.percentile(vals, 100 * alpha / 2)
    hi = np.percentile(vals, 100 * (1 - alpha / 2))
    
    return {"value": round(point, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4)}


def auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def lift_at_k(y_true, y_score, k=0.1):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    cut = max(1, int(np.ceil(k * len(y_true))))
    top = np.argsort(-y_score)[:cut]
    base = y_true.mean()
    return (y_true[top].mean() / base) if base > 0 else np.nan


def precision_at_k(y_true, y_score, k=0.1):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    cut = max(1, int(np.ceil(k * len(y_true))))
    return y_true[np.argsort(-y_score)[:cut]].mean()
