import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src.config import load_cfg, ensure_dirs

cfg = load_cfg()
ensure_dirs(cfg)

DATA = Path(cfg.paths.data_csv)
PRED = Path(cfg.paths.out_dir) / "predictions.csv"
MODEL = Path(cfg.paths.out_dir) / "champion_model.pkl"
FIGDIR = Path(cfg.paths.figs_dir)
FIGDIR.mkdir(parents=True, exist_ok=True)
TARGET = "bought_fragrance"

def plot_roc_pr_lift():
    df = pd.read_csv(PRED)
    y_true = df["y_true"].astype(int).to_numpy()
    y_proba = df["y_proba"].astype(float).to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "roc_curve.png", dpi=160)

    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure()
    plt.plot(rec, prec, label=f"PR AUC = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "pr_curve.png", dpi=160)

    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = max(int(cum_pos[-1]), 1)
    n = len(y_true)
    pct = np.arange(1, n + 1) / n
    baseline = total_pos / n
    denom = np.maximum(pct * n * baseline, 1e-9)
    lift = cum_pos / denom
    plt.figure()
    plt.plot(pct * 100, lift)
    plt.xlabel("% ranked by score")
    plt.ylabel("Lift vs baseline")
    plt.title("Cumulative Lift")
    plt.tight_layout()
    plt.savefig(FIGDIR / "lift_curve.png", dpi=160)

def get_feature_names(pre):
    num_cols = None
    cat_cols = None
    for name, trans, cols in pre.transformers_:
        if name == "num":
            num_cols = list(cols)
        if name == "cat":
            cat_cols = list(cols)
    cat_names = []
    if "cat" in pre.named_transformers_:
        ohe = pre.named_transformers_["cat"]
        if hasattr(ohe, "get_feature_names_out"):
            cat_names = list(ohe.get_feature_names_out(cat_cols))
    num_names = num_cols if num_cols is not None else []
    return num_names + cat_names

def plot_shap_summaries():
    if not MODEL.exists() or not DATA.exists():
        print("SHAP skipped: missing model or data")
        return

    pipe = joblib.load(MODEL)
    if "pre" not in pipe.named_steps or "clf" not in pipe.named_steps:
        print("SHAP skipped: pipeline missing pre or clf")
        return

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    df = pd.read_csv(DATA)
    X = df.drop(columns=[TARGET]) if TARGET in df.columns else df.copy()
    Xn = pre.transform(X)
    if hasattr(Xn, "toarray"):
        Xn = Xn.toarray()
    Xn = np.asarray(Xn, dtype=np.float32)
    feat_names = get_feature_names(pre)
    if len(feat_names) != Xn.shape[1]:
        feat_names = [f"f{i}" for i in range(Xn.shape[1])]
    rng = np.random.default_rng(42)
    bg_size = min(200, Xn.shape[0])
    background = Xn[rng.choice(Xn.shape[0], size=bg_size, replace=False)]
    samp_size = min(300, Xn.shape[0])
    X_shap = Xn[rng.choice(Xn.shape[0], size=samp_size, replace=False)]

    try:
        import shap
        try:
            expl = shap.TreeExplainer(
                clf,
                data=background,
                feature_perturbation="interventional",
                model_output="probability",
            )
            values = expl.shap_values(X_shap)
            data_for_plots = X_shap
        except Exception:
            expl = shap.Explainer(lambda z: clf.predict_proba(z)[:, 1], background)
            sv = expl(X_shap)
            if hasattr(sv, "values"):
                values = sv.values
                data_for_plots = sv.data if sv.data is not None else X_shap
            else:
                values = sv
                data_for_plots = X_shap

        if isinstance(values, list):
            values = values[1] if len(values) >= 2 else values[0]

        plt.figure()
        shap.summary_plot(values, data_for_plots, feature_names=feat_names, show=False)
        plt.tight_layout()
        plt.savefig(FIGDIR / "shap_summary_beeswarm.png", dpi=160)

        plt.figure()
        shap.summary_plot(values, data_for_plots, feature_names=feat_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(FIGDIR / "shap_summary_bar.png", dpi=160)

        mean_abs = np.mean(np.abs(values), axis=0)
        if mean_abs.size:
            top_idx = int(np.argsort(-mean_abs)[:1][0])
            plt.figure()
            shap.dependence_plot(top_idx, values, data_for_plots, feature_names=feat_names, show=False)
            plt.tight_layout()
            plt.savefig(FIGDIR / "shap_dependence_top.png", dpi=160)

        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        ).to_csv("reports/shap_top_features.csv", index=False)

    except Exception as e:
        print("SHAP skipped:", e)

if __name__ == "__main__":
    if PRED.exists():
        plot_roc_pr_lift()
    plot_shap_summaries()
    print("Wrote PNGs to", FIGDIR)
