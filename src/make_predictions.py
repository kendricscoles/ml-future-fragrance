import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.config import load_cfg, ensure_dirs

cfg = load_cfg()
ensure_dirs(cfg)

DATA = Path(cfg.paths.data_csv)
MODEL = Path(cfg.paths.out_dir) / "champion_model.pkl"
OUT = Path(cfg.paths.out_dir) / "predictions.csv"
TARGET = "bought_fragrance"

df = pd.read_csv(DATA)
y_true = df[TARGET].astype(int).to_numpy()
X = df.drop(columns=[TARGET])

model = joblib.load(MODEL)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X)[:, 1]
elif hasattr(model, "decision_function"):
    y_proba = model.decision_function(X)
    y_proba = MinMaxScaler().fit_transform(y_proba.reshape(-1, 1)).ravel()
else:
    y_proba = model.predict(X)

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"y_true": y_true, "y_proba": y_proba}).to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(y_true)} rows")
