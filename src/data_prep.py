import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

import argparse
import numpy as np
import pandas as pd

from src.config import load_cfg

cfg = load_cfg()


def clean_df(df):
    for col in df.columns:
        if df[col].dtype == object:
            s = df[col].astype(str).str.replace("[", "").str.replace("]", "").str.replace(",", ".").str.strip()
            try:
                df[col] = pd.to_numeric(s)
            except:
                pass
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def make_data(n=800, seed=42):
    rng = np.random.default_rng(seed)
    
    ages = ["18-24", "25-34", "35-44", "45-54", "55+"]
    regions = ["DE", "CH", "FR", "IT", "AT"]
    
    df = pd.DataFrame({
        "views_7d": rng.poisson(3, n),
        "add_to_cart_30d": rng.poisson(1.2, n),
        "orders_12m": rng.poisson(0.6, n),
        "avg_price_viewed": rng.normal(65, 20, n).clip(10, 200),
        "brand_diversity": rng.integers(1, 8, n),
        "days_since_last_purchase": rng.integers(0, 365, n),
        "campaign_clicks": rng.poisson(0.8, n),
        "age_group": rng.choice(ages, n, p=[0.18, 0.32, 0.22, 0.17, 0.11]),
        "region": rng.choice(regions, n, p=[0.35, 0.25, 0.18, 0.12, 0.10]),
    })
    
    # propensity score
    score = (0.45 * df["views_7d"] + 0.9 * df["add_to_cart_30d"] + 
             0.8 * df["campaign_clicks"] + 0.6 * df["orders_12m"] -
             0.003 * df["days_since_last_purchase"] + 
             0.01 * (df["avg_price_viewed"] - 60) + 
             0.05 * (df["brand_diversity"] - 3))
    
    age_fx = {"18-24": 0.05, "25-34": 0.08, "35-44": 0.03, "45-54": -0.02, "55+": -0.05}
    reg_fx = {"DE": 0.02, "CH": 0.06, "FR": 0.00, "IT": -0.01, "AT": 0.01}
    score += df["age_group"].map(age_fx) + df["region"].map(reg_fx)
    
    prob = 1 / (1 + np.exp(-(-2.0 + 0.25 * score)))
    df["bought_fragrance"] = (rng.random(n) < prob).astype(int)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/fragrance_data.csv")
    args = p.parse_args()

    df = clean_df(make_data(n=args.rows, seed=args.seed))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {df.shape}")


if __name__ == "__main__":
    main()
