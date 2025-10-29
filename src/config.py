from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Paths:
    data_csv: str
    out_dir: str
    predictions_csv: str
    metrics_json: str
    reports_dir: str
    figs_dir: str

@dataclass
class ModelCfg:
    estimator: str
    xgb: dict
    lgbm: dict | None = None

@dataclass
class Cfg:
    seed: int
    rows: int
    paths: Paths
    model: ModelCfg

def load_cfg(path: str = "config.yaml") -> Cfg:
    with open(path, "r") as f:
        c = yaml.safe_load(f)
    p = c["paths"]
    m = c["model"]
    return Cfg(
        seed=int(c["seed"]),
        rows=int(c["rows"]),
        paths=Paths(**p),
        model=ModelCfg(
            estimator=m["estimator"],
            xgb=m.get("xgb", {}),
            lgbm=m.get("lgbm", {}),
        ),
    )

def ensure_dirs(cfg: Cfg):
    Path(cfg.paths.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.reports_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.figs_dir).mkdir(parents=True, exist_ok=True)
