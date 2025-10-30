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
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba}).sort_values("y_proba", ascending=False).reset_index(drop=True)
    baseline = df["y_true"].mean()
    out = {}
    for p in top_percents:
        k = max(1, int(len(df) * p / 100))
        top = df.iloc[:k]
        top_rate = top["y_true"].mean()
        out[f"lift@{p}"] = float(top_rate / baseline) if baseline > 0 else float("nan")
    return out

def gains_by_decile(y_true, y_proba):
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df["rank"] = df["y_proba"].rank(method="first", ascending=False)
    df["decile"] = pd.qcut(df["rank"], 10, labels=False, duplicates="drop")
    g = (df.groupby("decile", as_index=False)
           .agg(positives=("y_true","sum"), count=("y_true","size")))
    g["rate"] = g["positives"] / g["count"]
    g = g.sort_values("decile", ascending=False)
    g["cum_positives"] = g["positives"].cumsum()
    g["cum_rate"] = g["cum_positives"] / g["count"].sum()
    return g

def selection_rate(y_true, y_pred):
    y_pred = np.asarray(y_pred).astype(int)
    return float(y_pred.mean())

def tpr(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    pos = (y_true == 1).sum()
    return float(tp / max(1, pos))

def ppv(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    pred_pos = (y_pred == 1).sum()
    return float(tp / max(1, pred_pos))

def main(pred_path: Path, data_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(pred_path)
    if "row_id" not in preds.columns:
        preds.insert(0, "row_id", range(len(preds)))
    if "y_proba" not in preds.columns and "y_score" in preds.columns:
        preds["y_proba"] = preds["y_score"]

    df = pd.read_csv(data_path)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    if "y_true" in preds.columns:
        merged = preds.merge(df[["row_id","bought_fragrance","age"]], on="row_id", how="left")
        y_true = merged["y_true"].fillna(merged["bought_fragrance"]).astype(int).to_numpy()
    else:
        merged = preds.merge(df[["row_id","bought_fragrance","age"]], on="row_id", how="left")
        y_true = merged["bought_fragrance"].astype(int).to_numpy()

    y_proba = merged["y_proba"].astype(float).to_numpy()
    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)

    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    lifts = lift_table(y_true, y_proba)

    g = gains_by_decile(y_true, y_proba)
    g.to_csv(outdir / "lift_by_decile.csv", index=False)

    if "age" in merged.columns:
        bins = [0,25,35,50,200]
        labels = ["<=25","26-35","36-50","50+"]
        merged["age_group"] = pd.cut(merged["age"], bins=bins, labels=labels, right=True, include_lowest=True)
        thresh = np.quantile(y_proba, 0.9)
        y_pred = (y_proba >= thresh).astype(int)
        rows = []
        for gname, d in merged.groupby("age_group"):
            if d.empty:
                continue
            sel = (d["y_proba"] >= thresh).astype(int)
            rows.append({
                "age_group": str(gname),
                "n": int(len(d)),
                "selection_rate": selection_rate(d["bought_fragrance"], sel),
                "tpr": tpr(d["bought_fragrance"], sel),
                "ppv": ppv(d["bought_fragrance"], sel)
            })
        if rows:
            fair = pd.DataFrame(rows)
            fair.to_csv(outdir / "fairness_age_group.csv", index=False)

    summary = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "lift@10": round(float(lifts.get("lift@10", np.nan)), 3) if pd.notna(lifts.get("lift@10", np.nan)) else None,
        "lift@20": round(float(lifts.get("lift@20", np.nan)), 3) if pd.notna(lifts.get("lift@20", np.nan)) else None,
        "lift@30": round(float(lifts.get("lift@30", np.nan)), 3) if pd.notna(lifts.get("lift@30", np.nan)) else None
    }
    pd.DataFrame([summary]).to_csv(outdir / "metrics_summary.csv", index=False)
    print("Metrics:", summary)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("reports"))
    args = ap.parse_args()
    main(args.pred, args.data, args.outdir)
