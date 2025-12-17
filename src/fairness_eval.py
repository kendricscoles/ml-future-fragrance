import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rate(y, p):
    return np.asarray(p, int).mean()

def tpr(y, p):
    yt, yp = np.asarray(y, int), np.asarray(p, int)
    return ((yt == 1) & (yp == 1)).sum() / max(1, (yt == 1).sum())

def ppv(y, p):
    yt, yp = np.asarray(y, int), np.asarray(p, int)
    return ((yt == 1) & (yp == 1)).sum() / max(1, (yp == 1).sum())

def fpr(y, p):
    yt, yp = np.asarray(y, int), np.asarray(p, int)
    return ((yt == 0) & (yp == 1)).sum() / max(1, (yt == 0).sum())


def fairness_metrics(df, y_true, y_pred, col):
    """compute metrics by group"""
    rows = []
    for g in df[col].unique():
        mask = df[col] == g
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        rows.append({
            "group": str(g), "n": int(mask.sum()),
            "selection_rate": float(rate(yt, yp)),
            "tpr": float(tpr(yt, yp)),
            "fpr": float(fpr(yt, yp)),
            "ppv": float(ppv(yt, yp)),
        })
    
    mdf = pd.DataFrame(rows)
    sr_gap = mdf["selection_rate"].max() - mdf["selection_rate"].min()
    tpr_gap = mdf["tpr"].max() - mdf["tpr"].min()
    fpr_gap = mdf["fpr"].max() - mdf["fpr"].min()
    
    return {
        "group_metrics": mdf,
        "selection_rate_gap": float(sr_gap),
        "demographic_parity_gap": float(sr_gap),
        "equalized_odds_gap": float(max(tpr_gap, fpr_gap)),
        "tpr_gap": float(tpr_gap),
        "fpr_gap": float(fpr_gap),
        "ppv_gap": float(mdf["ppv"].max() - mdf["ppv"].min()),
    }


def tradeoff_curve(df, y_true, y_prob, col, path):
    results = []
    for t in np.linspace(0.1, 0.9, 9):
        yp = (y_prob >= np.quantile(y_prob, t)).astype(int)
        m = fairness_metrics(df, y_true, yp, col)
        results.append({
            "thresh_pct": t * 100,
            "acc": float((y_true == yp).mean()),
            "sr_gap": m["selection_rate_gap"],
            "eq_odds": m["equalized_odds_gap"],
        })
    
    rdf = pd.DataFrame(results)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Threshold %")
    ax1.set_ylabel("Gap", color="tab:red")
    ax1.plot(rdf["thresh_pct"], rdf["sr_gap"], "o-", color="tab:red", label="Dem. Parity")
    ax1.plot(rdf["thresh_pct"], rdf["eq_odds"], "s--", color="tab:orange", label="Eq. Odds")
    ax1.legend(loc="upper left")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(rdf["thresh_pct"], rdf["acc"], "^-", color="tab:blue")
    
    plt.title("Fairness-Accuracy Tradeoff", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return rdf


def write_narrative(m, gdf, path):
    """write fairness narrative markdown"""
    sorted_g = gdf.sort_values("selection_rate", ascending=False)
    hi, lo = sorted_g.iloc[0]["group"], sorted_g.iloc[-1]["group"]
    
    eq_note = ("Equitable error rates." if m["equalized_odds_gap"] < 0.15 
               else "Notable differences in error rates.")
    dp_note = ("Balanced selection." if m["demographic_parity_gap"] < 0.15 
               else "Differences in selection rates.")
    
    rows = "\n".join([f"| {r['group']} | {r['n']} | {r['selection_rate']:.3f} | {r['tpr']:.3f} | {r['fpr']:.3f} | {r['ppv']:.3f} |" 
                      for _, r in gdf.iterrows()])
    
    md = f"""# Fairness Analysis

## Key Metrics
| Metric | Value |
|--------|-------|
| Demographic Parity Gap | {m['demographic_parity_gap']:.3f} |
| Equalized Odds Gap | {m['equalized_odds_gap']:.3f} |
| TPR Gap | {m['tpr_gap']:.3f} |
| FPR Gap | {m['fpr_gap']:.3f} |

## By Group
| Group | N | Sel Rate | TPR | FPR | PPV |
|-------|---|----------|-----|-----|-----|
{rows}

## Notes
- Highest selection: {hi}, Lowest: {lo}
- {eq_note}
- {dp_note}
- Threshold: top 10%, Protected: age_group
"""
    with open(path, "w") as f:
        f.write(md)
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/fragrance_data.csv")
    p.add_argument("--pred", default="artifacts/predictions.csv")
    p.add_argument("--outdir", default="reports")
    args = p.parse_args()

    out = Path(args.outdir)
    figs = out / "figures"
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    df = pd.read_csv(args.data)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    preds = pd.read_csv(args.pred)
    if "row_id" not in preds.columns:
        preds.insert(0, "row_id", range(len(preds)))
    if "y_score" not in preds.columns and "y_proba" in preds.columns:
        preds["y_score"] = preds["y_proba"]

    df = df.merge(preds[["row_id", "y_score"]], on="row_id", how="inner")

    if "age_group" not in df.columns:
        df["age_group"] = pd.cut(df["age"], [0, 25, 35, 50, 200], 
                                  labels=["<=25", "26-35", "36-50", "50+"], include_lowest=True)

    y_true = df["bought_fragrance"].astype(int).values
    y_prob = df["y_score"].values
    thresh = float(np.quantile(y_prob, 0.9))
    y_pred = (y_prob >= thresh).astype(int)

    m = fairness_metrics(df, y_true, y_pred, "age_group")
    gdf = m["group_metrics"]
    
    gdf.to_csv(out / "fairness_age_group.csv", index=False)

    ax = gdf.set_index("group")[["selection_rate", "tpr", "fpr", "ppv"]].plot(kind="bar")
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(figs / "fairness_age.png", dpi=160)
    plt.close()

    tdf = tradeoff_curve(df, y_true, y_prob, "age_group", figs / "fairness_tradeoff.png")
    print(f"Wrote {figs / 'fairness_tradeoff.png'}")

    write_narrative(m, gdf, out / "fairness_narrative.md")

    # update metrics.json
    mp = Path("artifacts/metrics.json")
    mj = json.load(open(mp)) if mp.exists() else {}
    mj.setdefault("fairness", {}).update({
        "selection_rate_gap": m["selection_rate_gap"],
        "demographic_parity_gap": m["demographic_parity_gap"],
        "equalized_odds_gap": m["equalized_odds_gap"],
        "tpr_gap": m["tpr_gap"], "fpr_gap": m["fpr_gap"], "ppv_gap": m["ppv_gap"],
        "threshold": thresh
    })
    json.dump(mj, open(mp, "w"), indent=2)

    print(f"Wrote {out / 'fairness_age_group.csv'}")
    print(f"Fairness gaps: sr={m['selection_rate_gap']:.3f}, tpr={m['tpr_gap']:.3f}")


if __name__ == "__main__":
    main()
