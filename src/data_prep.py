import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
import argparse
import numpy as np
import pandas as pd
import os
from src.config import load_cfg

cfg = load_cfg()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            s = (
                df[col].astype(str)
                .str.replace("[", "", regex=False)
                .str.replace("]", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            try:
                df[col] = pd.to_numeric(s)
            except Exception:
                pass
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def make_data(n=800, seed=42):
    rng = np.random.default_rng(seed)
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    regions = ["DE", "CH", "FR", "IT", "AT"]
    df = pd.DataFrame({
        "views_7d": rng.poisson(3, size=n),
        "add_to_cart_30d": rng.poisson(1.2, size=n),
        "orders_12m": rng.poisson(0.6, size=n),
        "avg_price_viewed": rng.normal(65, 20, size=n).clip(10, 200),
        "brand_diversity": rng.integers(1, 8, size=n),
        "days_since_last_purchase": rng.integers(0, 365, size=n),
        "campaign_clicks": rng.poisson(0.8, size=n),
        "age_group": rng.choice(age_groups, size=n, p=[0.18, 0.32, 0.22, 0.17, 0.11]),
        "region": rng.choice(regions, size=n, p=[0.35, 0.25, 0.18, 0.12, 0.10]),
    })
    score = (
        0.45 * df["views_7d"]
        + 0.9 * df["add_to_cart_30d"]
        + 0.8 * df["campaign_clicks"]
        + 0.6 * df["orders_12m"]
        - 0.003 * df["days_since_last_purchase"]
        + 0.01 * (df["avg_price_viewed"] - 60)
        + 0.05 * (df["brand_diversity"] - 3)
    )
    age_map = {"18-24": 0.05, "25-34": 0.08, "35-44": 0.03, "45-54": -0.02, "55+": -0.05}
    reg_map = {"DE": 0.02, "CH": 0.06, "FR": 0.00, "IT": -0.01, "AT": 0.01}
    score += df["age_group"].map(age_map) + df["region"].map(reg_map)
    prob = 1 / (1 + np.exp(-(-2.0 + 0.25 * score)))
    df["bought_fragrance"] = (rng.random(n) < prob).astype(int)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/fragrance_data.csv")
    args = parser.parse_args()

    df = make_data(n=args.rows, seed=args.seed)
    df = clean_dataframe(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
