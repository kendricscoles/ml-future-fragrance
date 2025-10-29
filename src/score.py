import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--target", default="bought_fragrance")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    df = pd.read_csv(args.data)

    y = None
    if args.target in df.columns:
        y = df[args.target].astype(int).to_numpy()
        X = df.drop(columns=[args.target])
    else:
        X = df

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        proba = (s - s.min()) / (s.max() - s.min() + 1e-9)
    else:
        proba = model.predict(X).astype(float)

    pred = (proba >= np.quantile(proba, 0.9)).astype(int)

    out_pred = out / "predictions.csv"
    out_top = out / "top_decile.csv"

    res = X.copy()
    res.insert(0, "row_id", np.arange(len(X)))
    res["score"] = proba
    if y is not None:
        res["y_true"] = y
    res.to_csv(out_pred, index=False)

    top_idx = np.argsort(-proba)[:max(1, int(0.1 * len(proba)))]
    res.iloc[top_idx].to_csv(out_top, index=False)

    print(json.dumps({"predictions": str(out_pred), "top_decile": str(out_top)}, indent=2))

if __name__ == "__main__":
    main()