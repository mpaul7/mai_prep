"""
Pandas Analysis Example: aggregating, filtering, summarizing.

Topics:
- Row and column filtering
- Derived columns
- GroupBy aggregations
- Multi-index aggregations and sorting
- Window functions (rolling and expanding)
- Cross-tabulations/pivot tables
- Percent change and contribution analysis
- Ranking and top-N per group
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_data(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num_rows = 500
    categories = ["electronics", "grocery", "home", "toys"]
    regions = ["north", "south", "east", "west"]

    df = pd.DataFrame(
        {
            "order_id": np.arange(1, num_rows + 1),
            "region": rng.choice(regions, size=num_rows),
            "category": rng.choice(categories, size=num_rows),
            "units": rng.integers(1, 10, size=num_rows),
            "price": np.round(rng.normal(50, 15, size=num_rows).clip(5, None), 2),
            "week": rng.integers(1, 53, size=num_rows),
        }
    )
    df["revenue"] = df["units"] * df["price"]
    return df


def run_analysis() -> None:
    df = generate_data()
    print("Data shape:", df.shape)
    print(df)

    # Filtering and derived columns
    high_price = df[df["price"] >= 60].copy()
    high_price["revenue_flag"] = high_price["revenue"] >= high_price["revenue"].median()
    print("High price rows:", len(high_price))
    print(high_price)

    # Aggregations
    by_region = (
        df.groupby("region", as_index=False)[["units", "revenue"]].sum().sort_values("revenue", ascending=False)
    )
    print("Revenue by region:\n", by_region.head())

    # Multi-index aggregations
    by_region_cat = (
        df.groupby(["region", "category"])  # multi-index
        .agg(revenue_sum=("revenue", "sum"), units_avg=("units", "mean"))
        .sort_values("revenue_sum", ascending=False)
    )
    print("\nRevenue by region-category (top 5):\n", by_region_cat.head())

    # Window functions: rolling weekly revenue per region
    df_sorted = df.sort_values(["region", "week", "order_id"])  # stable ordering
    df_sorted["weekly_rev"] = df_sorted.groupby(["region", "week"])['revenue'].transform('sum')
    df_sorted["rolling_rev_4w"] = (
        df_sorted.sort_values(["region", "week"]).groupby("region")["weekly_rev"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    )
    print("\nRolling 4-week revenue by region (sample):\n", df_sorted[["region", "week", "rolling_rev_4w"]].head(10))

    # Crosstab / pivot
    pivot = pd.pivot_table(
        df,
        index="region",
        columns="category",
        values="revenue",
        aggfunc="sum",
        fill_value=0.0,
    )
    print("\nRevenue pivot (region x category):\n", pivot)

    # Contribution analysis: share of category within region
    pivot_share = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
    print("\nRevenue share by category within region:\n", (pivot_share * 100).round(1))

    # Ranking top-N categories per region by revenue
    by_region_cat = by_region_cat.reset_index()
    by_region_cat["rank"] = by_region_cat.groupby("region")["revenue_sum"].rank(ascending=False, method="first")
    top2 = by_region_cat[by_region_cat["rank"] <= 2].sort_values(["region", "rank"])  # top 2 per region
    print("\nTop 2 categories per region:\n", top2)

    # Export summaries
    out_dir = "./results"
    by_region.to_csv(f"{out_dir}/analysis_by_region.csv", index=False)
    by_region_cat.to_csv(f"{out_dir}/analysis_by_region_category.csv", index=False)
    pivot.to_csv(f"{out_dir}/analysis_pivot_region_category.csv")
    pivot_share.to_csv(f"{out_dir}/analysis_pivot_share_region_category.csv")


if __name__ == "__main__":
    run_analysis()


