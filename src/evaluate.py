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


def lift_table(y_true, y_proba, top_pct=(10, 20, 30)):
    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p", ascending=False)
    base = df["y"].mean()
    out = {}
    for p in top_pct:
        k = max(1, int(len(df) * p / 100))
        rate = df.iloc[:k]["y"].mean()
        out[f"lift@{p}"] = float(rate / base) if base > 0 else float("nan")
    return out


def gains_by_decile(y_true, y_proba):
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["rank"] = df["p"].rank(method="first", ascending=False)
    df["dec"] = pd.qcut(df["rank"], 10, labels=False, duplicates="drop")
    g = df.groupby("dec", as_index=False).agg(pos=("y", "sum"), n=("y", "size"))
    g["rate"] = g["pos"] / g["n"]
    g = g.sort_values("dec", ascending=False)
    g["cum_pos"] = g["pos"].cumsum()
    g["cum_rate"] = g["cum_pos"] / g["n"].sum()
    return g


def sel_rate(y_true, y_pred):
    return float(np.asarray(y_pred).mean())

def tpr(y_true, y_pred):
    yt, yp = np.asarray(y_true, int), np.asarray(y_pred, int)
    tp = ((yt == 1) & (yp == 1)).sum()
    return float(tp / max(1, (yt == 1).sum()))

def ppv(y_true, y_pred):
    yt, yp = np.asarray(y_true, int), np.asarray(y_pred, int)
    tp = ((yt == 1) & (yp == 1)).sum()
    return float(tp / max(1, (yp == 1).sum()))


def main(pred_path, data_path, outdir):
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

    cols = ["row_id", "bought_fragrance"]
    if "age" in df.columns:
        cols.append("age")
    base = df[cols].copy()

    merged = preds.merge(base, on="row_id", how="left")

    if "y_true" in preds.columns:
        y_true = merged["y_true"].fillna(merged["bought_fragrance"]).astype(int).values
    else:
        y_true = merged["bought_fragrance"].astype(int).values

    y_proba = np.nan_to_num(merged["y_proba"].values, nan=0.0)

    auc = float(roc_auc_score(y_true, y_proba))
    pr = float(average_precision_score(y_true, y_proba))
    lifts = lift_table(y_true, y_proba)

    gains_by_decile(y_true, y_proba).to_csv(outdir / "lift_by_decile.csv", index=False)

    # fairness by age
    if "age" in df.columns:
        bins = [0, 25, 35, 50, 200]
        labels = ["<=25", "26-35", "36-50", "50+"]
        merged["age_group"] = pd.cut(merged["age"], bins=bins, labels=labels, include_lowest=True)
        thresh = float(np.quantile(y_proba, 0.9))
        rows = []
        for g, d in merged.groupby("age_group"):
            if len(d) == 0:
                continue
            sel = (d["y_proba"] >= thresh).astype(int)
            rows.append({
                "age_group": str(g),
                "n": len(d),
                "selection_rate": sel_rate(d["bought_fragrance"], sel),
                "tpr": tpr(d["bought_fragrance"], sel),
                "ppv": ppv(d["bought_fragrance"], sel)
            })
        if rows:
            pd.DataFrame(rows).to_csv(outdir / "fairness_age_group.csv", index=False)

    summary = {
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr, 4),
        "lift@10": round(lifts.get("lift@10", 0), 3),
        "lift@20": round(lifts.get("lift@20", 0), 3),
        "lift@30": round(lifts.get("lift@30", 0), 3),
    }
    pd.DataFrame([summary]).to_csv(outdir / "metrics_summary.csv", index=False)
    print("Metrics:", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("reports"))
    args = p.parse_args()
    main(args.pred, args.data, args.outdir)
