import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.config import load_cfg, ensure_dirs

cfg = load_cfg()
ensure_dirs(cfg)

def lift_table(y_true, y_proba, top_percents=(10,20,30)):
    df = pd.DataFrame({"y_true":y_true, "y_proba":y_proba}).sort_values("y_proba", ascending=False).reset_index(drop=True)
    baseline = df["y_true"].mean()
    out = {}
    for p in top_percents:
        k = max(1, int(len(df)*p/100))
        top = df.iloc[:k]
        top_rate = top["y_true"].mean()
        out[f"lift@{p}"] = float(top_rate / baseline) if baseline > 0 else float("nan")
    return out

def gains_by_decile(y_true, y_proba):
    df = pd.DataFrame({"y_true":y_true, "y_proba":y_proba})
    df["decile"] = pd.qcut(df["y_proba"].rank(method="first"), 10, labels=False, duplicates="drop")
    g = (df.groupby("decile", as_index=False)
           .agg(positives=("y_true","sum"), count=("y_true","size"))
           .sort_values("decile", ascending=False))
    g["rate"] = g["positives"]/g["count"]
    g["cum_positives"] = g["positives"].cumsum()
    g["cum_rate"] = g["cum_positives"]/g["count"].sum()
    return g

def main(pred_path:Path, data_path:Path, outdir:Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(pred_path)
    y_true = preds["y_true"].astype(int).to_numpy()
    y_proba = preds["y_proba"].astype(float).to_numpy()
    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)

    auc = float(roc_auc_score(y_true, y_proba))
    ap = float(average_precision_score(y_true, y_proba))
    lifts = lift_table(y_true, y_proba)

    g = gains_by_decile(y_true, y_proba)
    g.to_csv(outdir / "lift_by_decile.csv", index=False)

    try:
        df_full = pd.read_csv(data_path)
        if "age_group" in df_full.columns:
            df_scores = pd.DataFrame({"y_true": y_true, "y_proba": y_proba}).sort_values("y_proba", ascending=False).reset_index(drop=True)
            df_full = df_full.iloc[:len(df_scores)].copy()
            fairness = (pd.concat([df_full["age_group"], df_scores], axis=1)
                          .groupby("age_group")
                          .agg(pos_rate=("y_true","mean"), avg_score=("y_proba","mean"))
                          .reset_index())
            fairness.to_csv(outdir / "fairness_age_group.csv", index=False)
    except Exception as e:
        print("Fairness slice skipped:", e)

    summary = {
        "roc_auc": round(auc,4),
        "pr_auc": round(ap,4),
        "lift@10": round(float(lifts.get("lift@10", np.nan)),3) if lifts.get("lift@10", np.nan)==lifts.get("lift@10", np.nan) else None,
        "lift@20": round(float(lifts.get("lift@20", np.nan)),3) if lifts.get("lift@20", np.nan)==lifts.get("lift@20", np.nan) else None,
        "lift@30": round(float(lifts.get("lift@30", np.nan)),3) if lifts.get("lift@30", np.nan)==lifts.get("lift@30", np.nan) else None
    }
    pd.Series(summary).to_csv(outdir / "metrics_summary.csv")
    print("Metrics:", summary)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("reports"))
    args = ap.parse_args()
    main(args.pred, args.data, args.outdir)
