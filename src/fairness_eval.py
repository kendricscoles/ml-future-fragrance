import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rate(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return y_pred.mean()

def tpr(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    pos = (y_true == 1).sum()
    return tp / max(1, pos)

def ppv(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    pred_pos = (y_pred == 1).sum()
    return tp / max(1, pred_pos)

def main():
    data_path = Path("data/fragrance_data.csv")
    pred_path = Path("artifacts/predictions.csv")
    fig_dir = Path("reports/figures")
    out_dir = Path("reports")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    preds = pd.read_csv(pred_path)
    if "row_id" not in preds.columns:
        preds.insert(0, "row_id", range(len(preds)))
    if "y_score" not in preds.columns and "y_proba" in preds.columns:
        preds["y_score"] = preds["y_proba"]

    df = df.merge(preds[["row_id", "y_score"]], on="row_id", how="inner")

    if "age_group" not in df.columns:
        bins = [0, 25, 35, 50, 200]
        labels = ["<=25", "26-35", "36-50", "50+"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True, include_lowest=True)

    y_true = df["bought_fragrance"].astype(int)
    y_prob = df["y_score"].astype(float)
    thresh = float(y_prob.quantile(0.9))
    y_pred = (y_prob >= thresh).astype(int)

    rows = []
    for g, d in df.groupby("age_group"):
        sel = (d["y_score"] >= thresh).astype(int)
        rows.append({
            "age_group": str(g),
            "n": int(len(d)),
            "selection_rate": float(rate(d["bought_fragrance"], sel)),
            "tpr": float(tpr(d["bought_fragrance"], sel)),
            "ppv": float(ppv(d["bought_fragrance"], sel)),
        })
    out = pd.DataFrame(rows)
    sr_gap = float(out["selection_rate"].max() - out["selection_rate"].min())
    tpr_gap = float(out["tpr"].max() - out["tpr"].min())
    ppv_gap = float(out["ppv"].max() - out["ppv"].min())
    out.to_csv(out_dir / "fairness_age_group.csv", index=False)

    ax = out.set_index("age_group")[["selection_rate", "tpr", "ppv"]].plot(kind="bar")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(fig_dir / "fairness_age.png", dpi=160)
    plt.close(fig)

    mpath = Path("artifacts/metrics.json")
    if mpath.exists():
        with mpath.open() as f:
            m = json.load(f)
    else:
        m = {}
    m.setdefault("fairness", {})
    m["fairness"].update({
        "selection_rate_gap": sr_gap,
        "tpr_gap": tpr_gap,
        "ppv_gap": ppv_gap,
        "threshold": thresh
    })
    with mpath.open("w") as f:
        json.dump(m, f, indent=2)

if __name__ == "__main__":
    main()
