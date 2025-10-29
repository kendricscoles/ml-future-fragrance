import argparse
import warnings
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded")
warnings.filterwarnings("ignore", category=FutureWarning)

def load_feature_spec(model_path: Path):
    spec_path = model_path.with_name("feature_spec.json")
    if spec_path.exists():
        with open(spec_path, "r") as f:
            return json.load(f)
    return None

def build_tree_explainer_or_fallback(est, background_float):
    try:
        return shap.TreeExplainer(est, data=background_float, feature_perturbation="interventional", model_output="probability")
    except Exception:
        if hasattr(est, "predict_proba"):
            predict_fn = lambda X: est.predict_proba(X)[:, 1]
        elif hasattr(est, "decision_function"):
            predict_fn = lambda X: est.decision_function(X)
        else:
            predict_fn = lambda X: est.predict(X)
        masker = shap.maskers.Independent(background_float)
        return shap.Explainer(predict_fn, masker)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--samples", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    est = joblib.load(model_path)
    spec = load_feature_spec(model_path)

    df = pd.read_csv(args.data)
    for cand in ["bought_fragrance", "target", "label", "y"]:
        if cand in df.columns:
            df = df.drop(columns=[cand])

    est_name = spec.get("estimator") if spec else None
    feature_names = spec.get("feature_names") if spec else None
    cat_idx = spec.get("cat_idx") if spec else None

    use_cat = False
    try:
        from catboost import CatBoostClassifier, Pool
        use_cat = isinstance(est, CatBoostClassifier) or est_name == "cat"
    except Exception:
        use_cat = est_name == "cat"

    if use_cat:
        X_all = df.copy()
        if feature_names is not None:
            X_all = X_all.reindex(columns=feature_names)
        A = X_all
        background = A.to_numpy()
        background_float = background.astype(float, copy=False)
        explainer = build_tree_explainer_or_fallback(est, background_float)
        n = min(args.samples, len(A))
        rng = np.random.default_rng(17)
        idx = rng.choice(len(A), size=n, replace=False)
        X_sample = A.iloc[idx]
        X_sample_np = X_sample.to_numpy(dtype=float, copy=False)
        shap_values = explainer(X_sample_np)
        values = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
    else:
        X_all = pd.get_dummies(df, drop_first=False)
        if feature_names is not None:
            X_all = X_all.reindex(columns=feature_names, fill_value=0)
        X_all = X_all.astype(float)
        A = X_all
        background_float = A.to_numpy(dtype=float, copy=False)
        explainer = build_tree_explainer_or_fallback(est, background_float)
        n = min(args.samples, len(A))
        rng = np.random.default_rng(17)
        idx = rng.choice(len(A), size=n, replace=False)
        X_sample = A.iloc[idx]
        X_sample_np = X_sample.to_numpy(dtype=float, copy=False)
        shap_values = explainer(X_sample_np)
        values = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)

    plt.figure()
    shap.summary_plot(values, X_sample, show=False)
    bees = out_dir / "shap_summary_beeswarm.png"
    plt.savefig(bees, bbox_inches="tight", dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(values, X_sample, plot_type="bar", show=False)
    bar = out_dir / "shap_summary_bar.png"
    plt.savefig(bar, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"Saved SHAP figures to {bees} and {bar}")

if __name__ == "__main__":
    main()