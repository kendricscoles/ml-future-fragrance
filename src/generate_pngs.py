import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src.config import load_cfg, ensure_dirs

cfg = load_cfg()
ensure_dirs(cfg)

def plot_roc_pr_lift(pred_path: Path, out_dir: Path):
    preds = pd.read_csv(pred_path)
    if "y_proba" in preds.columns:
        y_score = preds["y_proba"].astype(float).to_numpy()
    elif "y_score" in preds.columns:
        y_score = preds["y_score"].astype(float).to_numpy()
    else:
        raise ValueError("Predictions must contain y_proba or y_score")
    if "y_true" not in preds.columns:
        raise ValueError("Predictions must contain y_true to plot curves")
    y_true = preds["y_true"].astype(int).to_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP = {ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=160)
    plt.close()

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = max(1, int(cum_pos[-1]))
    n = len(y_true)
    x = np.arange(1, n + 1) / n
    lift = (cum_pos / total_pos) / x
    plt.figure()
    plt.plot(x, lift, linewidth=2)
    plt.xlabel("Proportion of Population (ranked by score)")
    plt.ylabel("Lift")
    plt.title("Lift Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "lift_curve.png", dpi=160)
    plt.close()

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

def plot_shap(model_path: Path, data_path: Path, figs_dir: Path, reports_dir: Path, target_col: str = "bought_fragrance"):
    if not model_path.exists() or not data_path.exists():
        print("SHAP skipped: missing model or data")
        return
    try:
        import shap
    except Exception as e:
        print("SHAP skipped:", e)
        return
    pipe = joblib.load(model_path)
    if not hasattr(pipe, "named_steps") or "pre" not in pipe.named_steps or "clf" not in pipe.named_steps:
        print("SHAP skipped: pipeline missing pre or clf")
        return
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
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
        expl = shap.TreeExplainer(clf, data=background, feature_perturbation="interventional", model_output="probability")
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
    figs_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(values, data_for_plots, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(figs_dir / "shap_summary_beeswarm.png", dpi=160)
    plt.close()
    plt.figure()
    shap.summary_plot(values, data_for_plots, feature_names=feat_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(figs_dir / "shap_summary_bar.png", dpi=160)
    plt.close()
    mean_abs = np.mean(np.abs(values), axis=0)
    if mean_abs.size:
        top_idx = int(np.argsort(-mean_abs)[:1][0])
        plt.figure()
        shap.dependence_plot(top_idx, values, data_for_plots, feature_names=feat_names, show=False)
        plt.tight_layout()
        plt.savefig(figs_dir / "shap_dependence_top.png", dpi=160)
        plt.close()
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).to_csv(reports_dir / "shap_top_features.csv", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default=str(Path(cfg.paths.out_dir) / "champion_model.pkl"))
    ap.add_argument("--data", default=str(Path(cfg.paths.data_csv)))
    ap.add_argument("--with-shap", action="store_true")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    out_dir = Path(args.outdir)
    model_path = Path(args.model)
    data_path = Path(args.data)
    reports_dir = out_dir.parent

    plot_roc_pr_lift(pred_path, out_dir)
    if args.with_shap:
        plot_shap(model_path, data_path, out_dir, reports_dir)
    print("Wrote PNGs to", out_dir)

if __name__ == "__main__":
    main()
