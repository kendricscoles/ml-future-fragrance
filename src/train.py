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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import load_cfg, ensure_dirs

cfg = load_cfg()
ensure_dirs(cfg)

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

def load_data(csv_path: str, target: str):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}.")
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target])
    return X, y

def choose_estimator(kind: str, ci_mode: bool = False):
    kind = (kind or "xgb").lower()
    if kind in {"xgb", "xgboost"}:
        from xgboost import XGBClassifier
        params = dict(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=2,
            eval_metric="auc",
            n_jobs=-1,
            tree_method="hist",
            random_state=42,
        )
        if ci_mode:
            params.update(n_estimators=80, max_depth=4, subsample=0.7, colsample_bytree=0.7, n_jobs=2)
        return XGBClassifier(**params)
    elif kind in {"lgbm", "lightgbm"}:
        from lightgbm import LGBMClassifier
        params = dict(
            n_estimators=800,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            reg_lambda=0.0,
            reg_alpha=0.0,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
        )
        if ci_mode:
            params.update(n_estimators=120, subsample=0.7, colsample_bytree=0.7, n_jobs=2)
        return LGBMClassifier(**params)
    elif kind in {"logreg", "lr", "logistic"}:
        return LogisticRegression(max_iter=3000, solver="saga", n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Unknown estimator kind: {kind}")

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return pre

def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.1) -> float:
    n = len(y_true)
    k_n = max(1, int(np.ceil(k * n)))
    order = np.argsort(-y_score)
    top_k = order[:k_n]
    precision_at_k = y_true[top_k].mean() if k_n > 0 else 0.0
    baseline = y_true.mean() if y_true.mean() > 0 else 1e-9
    return float(precision_at_k / baseline)

def maybe_set_scale_pos_weight(model: Pipeline, y_train: np.ndarray):
    try:
        from xgboost import XGBClassifier
        is_xgb = isinstance(model.named_steps.get("clf"), XGBClassifier)
    except Exception:
        is_xgb = False
    if is_xgb:
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        spw = max(1.0, neg / pos)
        model.named_steps["clf"].set_params(scale_pos_weight=spw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target", default="bought_fragrance")
    parser.add_argument("--estimator", default="xgb")
    parser.add_argument("--ci-mode", "--ci_mode", dest="ci_mode", action="store_true",
                        default=os.getenv("CI", "0") == "1")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_data(args.data, args.target)
    pre = None
    if args.model:
        pipe_or_pre = joblib.load(args.model)
        if isinstance(pipe_or_pre, Pipeline):
            if "pre" in pipe_or_pre.named_steps:
                pre = pipe_or_pre.named_steps["pre"]
            elif len(pipe_or_pre) > 1:
                pre = pipe_or_pre[:-1]
        if pre is None:
            pre = getattr(pipe_or_pre, "transform", None) and pipe_or_pre
    if pre is None:
        pre = make_preprocessor(X)
    clf = choose_estimator(args.estimator, ci_mode=args.ci_mode)
    model = Pipeline([("pre", pre), ("clf", clf)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    maybe_set_scale_pos_weight(model, y_train)
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        y_score_val = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score_val = model.decision_function(X_test)
    else:
        y_score_val = model.predict(X_test)
    auc = float(roc_auc_score(y_test, y_score_val))
    lift10 = float(lift_at_k(y_test, y_score_val, k=0.10))
    pr_auc = float(average_precision_score(y_test, y_score_val))
    metrics = {"test": {"auc": auc, "pr_auc": pr_auc, "lift_at_10": lift10}}
    model_path = out_dir / "champion_model.pkl"
    joblib.dump(model, model_path)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved_model={model_path}")
    print(f"saved_metrics={metrics_path}")

if __name__ == "__main__":
    main()
