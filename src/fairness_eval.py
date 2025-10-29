import pandas as pd
import numpy as np

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
    df = pd.read_csv("data/fragrance_data.csv")
    preds = pd.read_csv("artifacts/predictions.csv")
    df = df.join(preds.set_index("row_id"), on="row_id")

    if "age_group" not in df.columns:
        bins = [0,25,35,50,200]
        labels = ["<=25","26-35","36-50","50+"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True, include_lowest=True)

    y_true = df["bought_fragrance"].astype(int)
    y_prob = df["y_score"].astype(float)
    thresh = df["y_score"].quantile(0.9)
    y_pred = (y_prob >= thresh).astype(int)

    groups = []
    for g, d in df.groupby("age_group"):
        groups.append({
            "age_group": str(g),
            "n": len(d),
            "selection_rate": rate(d["bought_fragrance"], (d["y_score"] >= thresh).astype(int)),
            "tpr": tpr(d["bought_fragrance"], (d["y_score"] >= thresh).astype(int)),
            "ppv": ppv(d["bought_fragrance"], (d["y_score"] >= thresh).astype(int)),
        })
    out = pd.DataFrame(groups)
    sr_gap = float(out["selection_rate"].max() - out["selection_rate"].min())
    tpr_gap = float(out["tpr"].max() - out["tpr"].min())
    ppv_gap = float(out["ppv"].max() - out["ppv"].min())
    out.to_csv("reports/fairness_age_group.csv", index=False)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = out.set_index("age_group")[["selection_rate","tpr","ppv"]].plot(kind="bar")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("reports/figures/fairness_age.png", dpi=160)
    plt.close()

    with open("artifacts/metrics.json","r") as f:
        import json
        m = json.load(f)
    m["fairness"] = {"selection_rate_gap": sr_gap, "tpr_gap": tpr_gap, "ppv_gap": ppv_gap, "threshold": float(thresh)}
    with open("artifacts/metrics.json","w") as f:
        json.dump(m, f, indent=2)

if __name__ == "__main__":
    main()
