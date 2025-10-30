import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from src.config import load_cfg, ensure_dirs

cfg = load_cfg()
ensure_dirs(cfg)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path(cfg.paths.data_csv)))
    p.add_argument("--model", default=str(Path(cfg.paths.out_dir) / "champion_model.pkl"))
    p.add_argument("--out", default=str(Path(cfg.paths.out_dir) / "predictions.csv"))
    p.add_argument("--target", default="bought_fragrance")
    p.add_argument("--index", default="")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    if args.index:
        idx = pd.read_csv(args.index)["row_id"].tolist()
        df = df.iloc[idx].copy()

    target = args.target
    if target in df.columns:
        y_true = df[target].astype(int).to_numpy()
        X = df.drop(columns=[target])
    else:
        y_true = None
        X = df.copy()

    pipe = joblib.load(args.model)
    if isinstance(pipe, Pipeline) and hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X)[:, 1]
    elif isinstance(pipe, Pipeline) and hasattr(pipe, "decision_function"):
        raw = pipe.decision_function(X)
        y_score = MinMaxScaler().fit_transform(np.asarray(raw).reshape(-1, 1)).ravel()
    else:
        y_score = pipe.predict(X)

    out = pd.DataFrame({"row_id": df["row_id"].values, "y_score": y_score})
    out["y_proba"] = out["y_score"]
    if y_true is not None:
        out["y_true"] = y_true
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows")

if __name__ == "__main__":
    main()
