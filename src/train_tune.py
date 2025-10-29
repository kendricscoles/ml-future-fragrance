import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from joblib import dump
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def detect_cats_nums(X):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return cat, num

def dummies_fit_transform(X_train, X_test):
    X_all = pd.get_dummies(pd.concat([X_train, X_test], axis=0, copy=False), drop_first=False)
    X_tr = X_all.iloc[: len(X_train)].copy()
    X_te = X_all.iloc[len(X_train) :].copy()
    return X_tr, X_te

def scale_fit_transform(X_train, X_test):
    scaler = StandardScaler(with_mean=False)
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_tr, X_te

def metrics_at_proba(y_true, proba, top_k=0.1):
    auc = float(roc_auc_score(y_true, proba))
    ap = float(average_precision_score(y_true, proba))
    n = len(proba)
    k = max(1, int(np.floor(top_k * n)))
    idx = np.argsort(proba)[::-1][:k]
    lift = (y_true[idx].mean() / y_true.mean()) if y_true.mean() > 0 else 0.0
    thr = float(sorted(proba, reverse=True)[k - 1])
    return auc, ap, float(lift), thr

def fit_logreg(X_tr, y_tr, n_iter, cv, rng):
    params = {"C": np.logspace(-3, 3, 100), "penalty": ["l2"], "solver": ["lbfgs"], "max_iter": [2000]}
    search = RandomizedSearchCV(
        LogisticRegression(class_weight="balanced"),
        param_distributions=params,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=rng,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_score_

def fit_xgb(X_tr, y_tr, n_iter, cv, rng, scale_pos_weight):
    from xgboost import XGBClassifier
    space = {
        "n_estimators": np.arange(150, 501),
        "max_depth": np.arange(3, 8),
        "learning_rate": np.linspace(0.02, 0.2, 40),
        "subsample": np.linspace(0.6, 1.0, 21),
        "colsample_bytree": np.linspace(0.6, 1.0, 21),
        "min_child_weight": np.arange(1, 8),
        "gamma": np.linspace(0.0, 2.0, 41),
    }
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cpu",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=rng,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_score_

def fit_lgbm(X_tr, y_tr, n_iter, cv, rng, scale_pos_weight):
    from lightgbm import LGBMClassifier
    space = {
        "n_estimators": np.arange(200, 701),
        "num_leaves": np.arange(15, 64),
        "learning_rate": np.linspace(0.02, 0.2, 40),
        "feature_fraction": np.linspace(0.6, 1.0, 21),
        "bagging_fraction": np.linspace(0.6, 1.0, 21),
        "min_child_samples": np.arange(5, 51),
        "reg_lambda": np.linspace(0.0, 5.0, 51),
    }
    base = LGBMClassifier(
        objective="binary",
        metric="auc",
        class_weight=None,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        verbose=-1,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=rng,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_score_

def fit_catboost(X_tr_df, y_tr, n_iter, cv, rng, cat_cols, class_weight_pos):
    from catboost import CatBoostClassifier
    cat_idx = [X_tr_df.columns.get_loc(c) for c in cat_cols]
    space = {
        "depth": np.arange(4, 9),
        "learning_rate": np.linspace(0.02, 0.2, 40),
        "l2_leaf_reg": np.linspace(1.0, 8.0, 71),
        "iterations": np.arange(300, 901),
        "border_count": np.arange(32, 257),
        "bagging_temperature": np.linspace(0.0, 5.0, 51),
    }
    base = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=int(rng.randint(0, 10_000)),
        verbose=False,
        class_weights=[1.0, float(class_weight_pos)],
        auto_class_weights=None,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=rng,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_tr_df, y_tr, cat_features=cat_idx)
    return search.best_estimator_, search.best_score_, cat_idx

def train_and_eval(estimator, X, y, n_iter, cv_splits, rng):
    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = float(neg / max(pos, 1))
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    if estimator in ["xgb", "lgbm", "logreg"]:
        Xtr, Xte = dummies_fit_transform(Xtr_raw, Xte_raw)
        if estimator == "logreg":
            Xtr, Xte = scale_fit_transform(Xtr, Xte)
    else:
        Xtr, Xte = Xtr_raw.copy(), Xte_raw.copy()
    feature_names = Xtr.columns.tolist()
    if estimator == "logreg":
        model, cv_best = fit_logreg(Xtr, ytr, n_iter, cv_splits, rng)
        p_te = model.predict_proba(Xte)[:, 1]
        cat_idx = None
    elif estimator == "xgb":
        model, cv_best = fit_xgb(Xtr, ytr, n_iter, cv_splits, rng, scale_pos_weight)
        p_te = model.predict_proba(Xte)[:, 1]
        cat_idx = None
    elif estimator == "lgbm":
        model, cv_best = fit_lgbm(Xtr, ytr, n_iter, cv_splits, rng, scale_pos_weight)
        p_te = model.predict_proba(Xte)[:, 1]
        cat_idx = None
    elif estimator == "cat":
        cat_cols, _ = detect_cats_nums(Xtr)
        model, cv_best, cat_idx = fit_catboost(Xtr, ytr, n_iter, cv_splits, rng, cat_cols, scale_pos_weight)
        from catboost import Pool
        pool_te = Pool(Xte, label=yte, cat_features=cat_idx)
        p_te = model.predict_proba(pool_te)[:, 1]
    else:
        raise ValueError("unsupported estimator")
    te_auc, te_ap, lift10, thr10 = metrics_at_proba(yte, p_te, top_k=0.1)
    te_auc2, te_ap2, lift20, thr20 = metrics_at_proba(yte, p_te, top_k=0.2)
    return {
        "model": model,
        "cv_auc": float(cv_best),
        "test_auc": float(te_auc),
        "test_ap": float(te_ap),
        "lift_at_10": float(lift10),
        "lift_at_20": float(lift20),
        "threshold_top_10": float(thr10),
        "threshold_top_20": float(thr20),
        "Xtr_shape": list(Xtr.shape),
        "Xte_shape": list(Xte.shape),
        "pos": pos,
        "neg": neg,
        "scale_pos_weight": scale_pos_weight,
        "estimator": estimator,
        "holdout_proba": p_te.tolist(),
        "holdout_y": yte.tolist(),
        "feature_names": feature_names,
        "cat_idx": cat_idx,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="bought_fragrance")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=60)
    ap.add_argument("--estimator", choices=["logreg", "xgb", "lgbm", "cat"], default="xgb")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise SystemExit("target column not found")
    y = df[args.target].astype(int).to_numpy()
    X = df.drop(columns=[args.target]).copy()
    rng = check_random_state(args.seed)
    results = train_and_eval(args.estimator, X, y, args.n_iter, args.cv, rng)
    model_path = out / "champion_model.pkl"
    dump(results["model"], model_path)
    preds_path = out / "predictions_holdout.csv"
    holdout = pd.DataFrame({"y_true": results["holdout_y"], "proba": results["holdout_proba"]})
    holdout.to_csv(preds_path, index=False)
    metrics = {
        "estimator": results["estimator"],
        "cv_auc": results["cv_auc"],
        "test_auc": results["test_auc"],
        "test_ap": results["test_ap"],
        "lift_at_10": results["lift_at_10"],
        "lift_at_20": results["lift_at_20"],
        "threshold_top_10": results["threshold_top_10"],
        "threshold_top_20": results["threshold_top_20"],
        "pos": results["pos"],
        "neg": results["neg"],
        "scale_pos_weight": results["scale_pos_weight"],
        "Xtr_shape": results["Xtr_shape"],
        "Xte_shape": results["Xte_shape"],
        "predictions_csv": str(preds_path),
        "model_path": str(model_path),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    feature_spec = {
        "estimator": results["estimator"],
        "feature_names": results["feature_names"],
        "cat_idx": results["cat_idx"],
    }
    with open(out / "feature_spec.json", "w") as f:
        json.dump(feature_spec, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()