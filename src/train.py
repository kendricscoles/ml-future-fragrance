import os, sys, random
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_cfg, ensure_dirs

cfg = load_cfg()
ensure_dirs(cfg)


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


def load_data(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"missing target: {target}")
    return df.drop(columns=[target]), df[target].astype(int).values


def choose_estimator(kind, ci_mode=False):
    kind = (kind or "xgb").lower()
    
    if kind in {"xgb", "xgboost"}:
        from xgboost import XGBClassifier
        params = dict(
            n_estimators=600, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, reg_alpha=0.0, min_child_weight=2,
            eval_metric="auc", n_jobs=-1, tree_method="hist", random_state=42,
        )
        if ci_mode:
            params.update(n_estimators=80, max_depth=4, subsample=0.7, 
                         colsample_bytree=0.7, n_jobs=2)
        return XGBClassifier(**params)
    
    elif kind in {"lgbm", "lightgbm"}:
        from lightgbm import LGBMClassifier
        params = dict(
            n_estimators=800, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, objective="binary",
            reg_lambda=0.0, reg_alpha=0.0, min_child_samples=20,
            random_state=42, n_jobs=-1,
        )
        if ci_mode:
            params.update(n_estimators=120, subsample=0.7, colsample_bytree=0.7, n_jobs=2)
        return LGBMClassifier(**params)
    
    elif kind in {"logreg", "lr", "logistic"}:
        return LogisticRegression(max_iter=3000, solver="saga", n_jobs=-1, random_state=42)
    
    raise ValueError(f"unknown estimator: {kind}")


def make_preprocessor(X):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def lift_at_k(y_true, y_score, k=0.1):
    n = len(y_true)
    cutoff = max(1, int(np.ceil(k * n)))
    top = np.argsort(-y_score)[:cutoff]
    baseline = y_true.mean() or 1e-9
    return float(y_true[top].mean() / baseline)


def run_cv(X, y, estimator, n_folds, ci_mode, out_dir):
    """k-fold CV for stability check"""
    from src.metrics import lift_at_k as m_lift
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    
    for i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        pre = make_preprocessor(X_tr)
        clf = choose_estimator(estimator, ci_mode=ci_mode)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        _set_class_weight(pipe, y_tr)
        pipe.fit(X_tr, y_tr)
        
        if hasattr(pipe, "predict_proba"):
            scores = pipe.predict_proba(X_val)[:, 1]
        elif hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(X_val)
        else:
            scores = pipe.predict(X_val)
        
        res = {
            "fold": i + 1,
            "auc": float(roc_auc_score(y_val, scores)),
            "pr_auc": float(average_precision_score(y_val, scores)),
            "lift_at_10": float(m_lift(y_val, scores, k=0.10)),
        }
        results.append(res)
        print(f"  Fold {i+1}/{n_folds}: AUC={res['auc']:.4f}, PR={res['pr_auc']:.4f}")
    
    # aggregate
    agg = {}
    for m in ["auc", "pr_auc", "lift_at_10"]:
        vals = [r[m] for r in results]
        agg[f"{m}_mean"] = round(np.mean(vals), 4)
        agg[f"{m}_std"] = round(np.std(vals), 4)
        agg[f"{m}_min"] = round(min(vals), 4)
        agg[f"{m}_max"] = round(max(vals), 4)
    
    cv_out = {"n_folds": n_folds, "folds": results, "aggregated": agg}
    
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(cv_out, f, indent=2)
    print(f"saved_cv_results={out_dir / 'cv_results.json'}")
    
    # stability csv
    rpt = Path("reports")
    rpt.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(rpt / "cv_stability.csv", index=False)
    print(f"saved_cv_stability={rpt / 'cv_stability.csv'}")
    
    print(f"\n=== CV Summary ({n_folds}-fold) ===")
    for m in ["auc", "pr_auc", "lift_at_10"]:
        print(f"  {m}: {agg[f'{m}_mean']:.4f} ± {agg[f'{m}_std']:.4f}")
    
    return cv_out


def _set_class_weight(pipe, y):
    try:
        from xgboost import XGBClassifier
        if isinstance(pipe.named_steps.get("clf"), XGBClassifier):
            pos = max(1, (y == 1).sum())
            neg = max(1, (y == 0).sum())
            pipe.named_steps["clf"].set_params(scale_pos_weight=max(1.0, neg/pos))
    except:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--target", default="bought_fragrance")
    p.add_argument("--estimator", default="xgb")
    p.add_argument("--ci-mode", "--ci_mode", dest="ci_mode", action="store_true",
                   default=os.getenv("CI", "0") == "1")
    p.add_argument("--cv-folds", "--cv_folds", dest="cv_folds", type=int, default=0)
    args = p.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    X, y = load_data(args.data, args.target)
    
    # optional cv
    if args.cv_folds > 0:
        print(f"\n=== Running {args.cv_folds}-fold Cross-Validation ===")
        run_cv(X, y, args.estimator, args.cv_folds, args.ci_mode, out_dir)
        print()
    
    # preprocessor
    pre = None
    if args.model:
        loaded = joblib.load(args.model)
        if isinstance(loaded, Pipeline):
            pre = loaded.named_steps.get("pre") or (loaded[:-1] if len(loaded) > 1 else None)
        if pre is None and hasattr(loaded, "transform"):
            pre = loaded
    if pre is None:
        pre = make_preprocessor(X)
    
    clf = choose_estimator(args.estimator, ci_mode=args.ci_mode)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=y)
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    
    _set_class_weight(pipe, y_tr)
    pipe.fit(X_tr, y_tr)
    
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_te)[:, 1]
    elif hasattr(pipe, "decision_function"):
        y_score = pipe.decision_function(X_te)
    else:
        y_score = pipe.predict(X_te)
    
    # bootstrap CIs
    from src.metrics import bootstrap_metric, lift_at_k as m_lift
    
    auc = bootstrap_metric(y_te, y_score, roc_auc_score, n_bootstrap=1000, ci_mode=args.ci_mode)
    pr = bootstrap_metric(y_te, y_score, average_precision_score, n_bootstrap=1000, ci_mode=args.ci_mode)
    lift = bootstrap_metric(y_te, y_score, lambda y,s: m_lift(y,s,k=0.10), n_bootstrap=1000, ci_mode=args.ci_mode)
    
    metrics = {"test": {
        "auc": auc["value"], "auc_ci_lower": auc["ci_lower"], "auc_ci_upper": auc["ci_upper"],
        "pr_auc": pr["value"], "pr_auc_ci_lower": pr["ci_lower"], "pr_auc_ci_upper": pr["ci_upper"],
        "lift_at_10": lift["value"], "lift_at_10_ci_lower": lift["ci_lower"], "lift_at_10_ci_upper": lift["ci_upper"],
    }}
    
    # save
    joblib.dump(pipe, out_dir / "champion_model.pkl")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"row_id": te_idx}).to_csv(out_dir / "test_index.csv", index=False)
    
    print(json.dumps(metrics, indent=2))
    print(f"saved_model={out_dir / 'champion_model.pkl'}")
    print(f"saved_metrics={out_dir / 'metrics.json'}")
    print(f"saved_test_index={out_dir / 'test_index.csv'}")


if __name__ == "__main__":
    main()
