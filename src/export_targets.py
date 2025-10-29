import pandas as pd
from pathlib import Path

def main():
    preds = pd.read_csv("artifacts/predictions_holdout.csv")
    preds["decile"] = pd.qcut(preds["proba"], q=10, labels=False, duplicates="drop") + 1

    lift = (
        preds.groupby("decile", as_index=False, observed=False)
        .agg({"y_true": ["mean", "count"]})
        .sort_values(("decile", ""), ascending=False)
    )

    preds_sorted = preds.sort_values("proba", ascending=False)
    top10 = preds_sorted.head(int(0.1 * len(preds_sorted)))
    top20 = preds_sorted.head(int(0.2 * len(preds_sorted)))

    Path("artifacts").mkdir(exist_ok=True)
    top10.to_csv("artifacts/target_top10.csv", index=False)
    top20.to_csv("artifacts/target_top20.csv", index=False)
    lift.to_csv("artifacts/lift_by_decile.csv", index=False)

    print("Wrote artifacts/target_top10.csv, artifacts/target_top20.csv, artifacts/lift_by_decile.csv")

if __name__ == "__main__":
    main()