"""
Basic statistics with pandas/numpy/scipy: correlation, covariance, probabilities.

Topics:
- Pearson and Spearman correlation
- Covariance matrix
- Z-scores and probability (normal CDF / survival)
- Binomial probabilities
- Simple conditional probability from contingency tables
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def generate_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.normal(0, 1, n)
    y = 0.6 * x + rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    cat = rng.choice(["A", "B"], p=[0.4, 0.6], size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z, "cat": cat})
    return df


def run_example() -> None:
    df = generate_data()
    print("Head:\n", df.head())

    # Correlation
    pearson_xy = df[["x", "y"]].corr(method="pearson").loc["x", "y"]
    spearman_xy = df[["x", "y"]].corr(method="spearman").loc["x", "y"]
    print("Pearson r(x,y):", round(pearson_xy, 3))
    print("Spearman rho(x,y):", round(spearman_xy, 3))

    # Covariance
    cov = df[["x", "y", "z"]].cov()
    print("\nCovariance matrix:\n", cov)

    # Z-scores and normal probabilities
    scores = stats.zscore(df["y"], nan_policy="omit")
    # Probability that a standard normal > 1.96
    p_gt_196 = stats.norm.sf(1.96)  # survival function 1 - CDF
    print("\nP(Z > 1.96):", round(float(p_gt_196), 5))

    # Binomial: probability of >= k successes
    n, p, k = 10, 0.3, 4
    p_ge_k = stats.binom.sf(k - 1, n=n, p=p)
    print(f"P(X >= {k}) for Bin(n={n}, p={p}):", round(float(p_ge_k), 5))

    # Conditional probability from contingency
    # Example: P(cat=A | y>0)
    df["y_pos"] = df["y"] > 0
    contingency = pd.crosstab(df["y_pos"], df["cat"])
    p_A_given_ypos = contingency.loc[True, "A"] / contingency.loc[True].sum()
    print("\nP(A | y>0):", round(float(p_A_given_ypos), 3))

    # Export a small summary
    summary = pd.DataFrame(
        {
            "metric": ["pearson_xy", "spearman_xy", "p_gt_196", "binom_ge_4", "p_A_given_ypos"],
            "value": [pearson_xy, spearman_xy, p_gt_196, p_ge_k, p_A_given_ypos],
        }
    )
    summary.to_csv("./results/statistics_summary.csv", index=False)


if __name__ == "__main__":
    run_example()


