"""Baseline comparisons for propensity model"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from src.metrics import lift_at_k


def random_baseline(y, seed=42):
    return np.random.RandomState(seed).random(len(y))


def heuristic_baseline(df, col="views_7d"):
    vals = df[col].values.astype(float)
    if vals.max() == vals.min():
        return np.full(len(vals), 0.5)
    return (vals - vals.min()) / (vals.max() - vals.min())


def baseline_metrics(y, scores, name):
    y = np.asarray(y, int)
    return {
        "model": name,
        "auc": round(roc_auc_score(y, scores), 4),
        "pr_auc": round(average_precision_score(y, scores), 4),
        "lift_at_10": round(lift_at_k(y, scores, 0.10), 4),
        "lift_at_20": round(lift_at_k(y, scores, 0.20), 4),
    }


def plot_comparison(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = {"Champion": "#2ecc71", "Random": "#e74c3c", "Heuristic (views_7d)": "#3498db"}
    
    for ax, (m, title) in zip(axes, [("auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("lift_at_10", "Lift@10")]):
        vals = df[m].tolist()
        bars = ax.bar(df["model"], vals, color=[colors.get(x, "#999") for x in df["model"]])
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        if m == "auc":
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        elif m == "lift_at_10":
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    
    plt.suptitle("Champion vs Baselines", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("reports"))
    p.add_argument("--target", default="bought_fragrance")
    p.add_argument("--heuristic-col", default="views_7d")
    args = p.parse_args()
    
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "figures").mkdir(exist_ok=True)
    
    df = pd.read_csv(args.data)
    preds = pd.read_csv(args.pred)
    
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))
    
    if "row_id" in preds.columns:
        merged = preds.merge(df[["row_id", args.target, args.heuristic_col]], on="row_id", how="left")
        y = merged[args.target].values
        heur_vals = merged[args.heuristic_col].values
    else:
        y = df[args.target].values[:len(preds)]
        heur_vals = df[args.heuristic_col].values[:len(preds)]
    
    champ = preds["y_proba"].values if "y_proba" in preds.columns else preds["y_score"].values
    rand = random_baseline(y)
    heur = heuristic_baseline(pd.DataFrame({args.heuristic_col: heur_vals}), args.heuristic_col)
    
    results = [
        baseline_metrics(y, champ, "Champion"),
        baseline_metrics(y, rand, "Random"),
        baseline_metrics(y, heur, "Heuristic (views_7d)"),
    ]
    
    cmp_df = pd.DataFrame(results)
    cmp_df.to_csv(args.outdir / "baseline_comparison.csv", index=False)
    print(f"Wrote {args.outdir / 'baseline_comparison.csv'}")
    print(cmp_df.to_string(index=False))
    
    plot_comparison(cmp_df, args.outdir / "figures" / "baseline_comparison.png")
    
    c_auc = cmp_df[cmp_df["model"] == "Champion"]["auc"].values[0]
    r_auc = cmp_df[cmp_df["model"] == "Random"]["auc"].values[0]
    print(f"\n=== Summary ===")
    print(f"Champion AUC improvement over random: +{c_auc - r_auc:.4f}")


if __name__ == "__main__":
    main()
