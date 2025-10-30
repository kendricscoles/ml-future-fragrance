import joblib, numpy as np, pandas as pd
from pathlib import Path
import shap, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def main():
    data_path = Path("data/fragrance_data.csv")
    model_path = Path("artifacts/champion_model.pkl")
    out_dir = Path("reports")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    y = df["bought_fragrance"].astype(int).to_numpy()
    X = df.drop(columns=["bought_fragrance"])

    pipe = joblib.load(model_path)
    if isinstance(pipe, Pipeline):
        model = pipe.named_steps.get("clf", pipe)
        X_trans = pipe.named_steps["pre"].fit_transform(X)
    else:
        model = pipe
        X_trans = X

    n = min(1500, X_trans.shape[0])
    idx = np.random.RandomState(42).choice(X_trans.shape[0], size=n, replace=False)
    X_small = X_trans[idx]

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_small)
    except Exception:
        explainer = shap.Explainer(model, X_small)
        sv = explainer(X_small)

    plt.figure()
    shap.plots.beeswarm(sv, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary_beeswarm.png", dpi=160)
    plt.close()

    if hasattr(sv, "values"):
        vals = np.abs(sv.values).mean(axis=0)
        imp = pd.DataFrame({"feature": [str(i) for i in range(len(vals))], "mean_abs_shap": vals})
        imp.sort_values("mean_abs_shap", ascending=False).to_csv(out_dir / "shap_top_features.csv", index=False)

        top_idx = np.argsort(vals)[::-1][:20]
        plt.figure()
        plt.bar(range(len(top_idx)), vals[top_idx])
        plt.xticks(range(len(top_idx)), [str(i) for i in top_idx], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "shap_summary_bar.png", dpi=160)
        plt.close()

if __name__ == "__main__":
    main()
