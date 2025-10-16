"""
Pandas Data Wrangling Example Script

This script demonstrates a realistic end-to-end data wrangling workflow commonly
assessed in data analyst scripting tests:

- Synthetic messy data generation
- Data loading and dtype control
- Missing value handling
- Duplicate detection and removal
- Type conversions and parsing
- Text cleaning and standardization
- Date handling and feature extraction
- Conditional transformations
- Aggregations and groupby
- Window functions
- Joins/merges across DataFrames
- Reshaping: melt/pivot, wide <-> long
- Outlier detection (IQR) and capping
- Export to CSV/Parquet

Run:
  python data_wrangling_example.py

Requires:
  - pandas
  - pyarrow (for Parquet export)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd


def generate_messy_data(random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate two related messy DataFrames: orders and customers.

    Returns:
        orders_df, customers_df
    """
    rng = np.random.default_rng(random_seed)

    num_customers = 50
    customer_ids = np.arange(1000, 1000 + num_customers)
    customer_names = [
        f"Customer {i}" if i % 7 != 0 else f" customer  {i}  " for i in range(num_customers)
    ]
    cities = [
        "New York",
        "new york",
        "San Francisco",
        "san  francisco",
        "Austin",
        None,
    ]
    states = ["NY", "CA", "TX", "CA", "NY", None]

    customers_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "name": customer_names,
            "city": rng.choice(cities, size=num_customers).astype(object),
            "state": rng.choice(states, size=num_customers).astype(object),
            # phone numbers with messy formatting
            "phone": rng.choice(
                [
                    "(555) 123-4567",
                    "555-123-4567",
                    "5551234567",
                    None,
                    "+1 555 123 4567",
                ],
                size=num_customers,
            ).astype(object),
        }
    )
    # introduce duplicate customer rows
    customers_df = pd.concat([customers_df, customers_df.iloc[[2, 5]]], ignore_index=True)
    

    num_orders = 200
    order_ids = np.arange(5000, 5000 + num_orders)
    order_customer_ids = rng.choice(customer_ids, size=num_orders)
    base_date = pd.Timestamp("2023-01-01")
    order_dates = base_date + pd.to_timedelta(rng.integers(0, 365, size=num_orders), unit="D")
    # strings with currency symbols, extra spaces, and occasional invalids
    prices = rng.normal(loc=100.0, scale=30.0, size=num_orders)
    prices = np.clip(prices, 5, None)
    price_texts = [
        f" $ {p:0.2f} " if i % 11 != 0 else "N/A" for i, p in enumerate(prices)
    ]
    quantities = rng.integers(1, 6, size=num_orders)

    # categories with inconsistent capitalization and whitespace
    categories = rng.choice(
        ["Electronics", " electronics", "GROCERY ", "grocery", "Home & Kitchen", "home  &  kitchen"],
        size=num_orders,
    )

    orders_df = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": order_customer_ids,
            "order_date": order_dates.astype(str),  # make it messy as strings
            "price": price_texts,
            "quantity": quantities,
            "category": categories,
        }
    )
    # add deliberate duplicate and missing values
    orders_df = pd.concat([orders_df, orders_df.iloc[[3]]], ignore_index=True)
    orders_df.loc[7, "quantity"] = None
    orders_df.loc[12, "order_date"] = "2023-13-40"  # invalid

    return orders_df, customers_df


def clean_text(s: pd.Series) -> pd.Series:
    """Trim, de-duplicate internal whitespace, and lowercase for standardization."""
    s = s.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.lower()
    return s


def parse_currency_to_float(s: pd.Series) -> pd.Series:
    """Parse messy currency strings like " $ 12.34 " or "N/A" to float."""
    s = s.astype("string").str.strip()
    s = s.str.replace(r"[^0-9.]+", "", regex=True)
    # Convert empty strings to NaN
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors="coerce")


def cap_outliers_iqr(values: pd.Series, iqr_multiplier: float = 1.5) -> pd.Series:
    """Cap outliers using IQR method; returns a new capped series."""
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    return values.clip(lower=lower, upper=upper)


def main() -> None:
    orders_raw, customers_raw = generate_messy_data()
    orders_raw.to_csv("./results/orders_raw.csv", index=False)
    customers_raw.to_csv("./results/customers_raw.csv", index=False)

    # # Load with explicit dtypes where possible
    orders = orders_raw.copy()
    customers = customers_raw.copy()

    # # Basic inspection
    print("Raw orders shape:", orders.shape)
    print("Raw customers shape:", customers.shape)

    # # Handle duplicates
    orders = orders.drop_duplicates()
    customers = customers.drop_duplicates()

    # # Parse price and dates
    orders["price_num"] = parse_currency_to_float(orders["price"])  # may yield NaN

    # Handle invalid or missing dates; coerce to NaT then fill with a sentinel and flag
    orders["order_date_parsed"] = pd.to_datetime(orders["order_date"], errors="coerce")
    orders["has_bad_date"] = orders["order_date_parsed"].isna()
    sentinel_date = pd.Timestamp("1970-01-01")
    orders["order_date_parsed"] = orders["order_date_parsed"].fillna(sentinel_date)

    # # Quantity: fill missing with median; ensure integer type
    median_qty = int(pd.Series(orders["quantity"]).dropna().median())

    orders["quantity_filled"] = pd.Series(orders["quantity"]).fillna(median_qty).astype(int)


    # # Standardize category text
    orders["category_std"] = clean_text(orders["category"]).replace(
        {
            "electronics": "electronics",
            "grocery": "grocery",
            "home & kitchen": "home & kitchen",
        }
    )

    # # Create revenue and cap extreme values
    orders["revenue"] = orders["price_num"] * orders["quantity_filled"]
    orders["revenue_capped"] = cap_outliers_iqr(orders["revenue"])  # avoid undue influence

    # # Date features
    orders["order_year"] = orders["order_date_parsed"].dt.year
    orders["order_month"] = orders["order_date_parsed"].dt.month
    orders["order_day"] = orders["order_date_parsed"].dt.day
    orders["order_dow"] = orders["order_date_parsed"].dt.day_name()

    # # Clean customers text fields
    customers["name_clean"] = clean_text(customers["name"])
    customers["city_clean"] = clean_text(customers["city"])
    customers["state_clean"] = clean_text(customers["state"])

    # # Normalize phone number to digits only
    customers["phone_digits"] = (
        customers["phone"].astype("string").str.replace(r"\D+", "", regex=True)
    )

    # # Simple imputation for city/state: fill missing with "unknown"
    customers["city_filled"] = customers["city_clean"].fillna("unknown")
    customers["state_filled"] = customers["state_clean"].fillna("unknown")

    # # Merge orders with customers
    fact_orders = orders.merge(
        customers[["customer_id", "name_clean", "city_filled", "state_filled", "phone_digits"]],
        how="left",
        on="customer_id",
        validate="many_to_one",
    )

    # # Aggregations: customer-level revenue
    customer_rev = (
        fact_orders.groupby("customer_id", as_index=False)
        .agg(total_revenue=("revenue_capped", "sum"), order_count=("order_id", "nunique"))
        .sort_values("total_revenue", ascending=False)
    )

    # # Window functions: running total per customer by date
    fact_orders = fact_orders.sort_values(["customer_id", "order_date_parsed", "order_id"])  # stable
    fact_orders["running_revenue"] = (
        fact_orders.groupby("customer_id")["revenue_capped"].cumsum()
    )


    # # Reshaping: monthly revenue by category pivoted wide
    monthly_cat = (
        fact_orders.groupby(["order_year", "order_month", "category_std"], as_index=False)
        .agg(monthly_revenue=("revenue_capped", "sum"))
    )
    monthly_pivot = monthly_cat.pivot_table(
        index=["order_year", "order_month"],
        columns="category_std",
        values="monthly_revenue",
        fill_value=0.0,
        aggfunc="sum",
    ).reset_index()

    # # Melt back to long format
    monthly_long = monthly_pivot.melt(
        id_vars=["order_year", "order_month"],
        var_name="category",
        value_name="monthly_revenue",
    )

    # # Conditional transformation example: high value flag
    revenue_threshold = fact_orders["revenue_capped"].quantile(0.9)
    fact_orders["is_high_value_order"] = fact_orders["revenue_capped"] >= revenue_threshold

    # # Show a few results
    print("Cleaned orders head:\n", fact_orders.head(3))
    print("\nTop 5 customers by revenue:\n", customer_rev.head(5))
    print("\nMonthly revenue by category (long):\n", monthly_long.head(6))

    # # Export examples
    out_dir = "./results"
    # Save CSVs
    fact_orders.to_csv(f"{out_dir}/fact_orders.csv", index=False)
    customer_rev.to_csv(f"{out_dir}/customer_revenue.csv", index=False)
    monthly_long.to_csv(f"{out_dir}/monthly_category_revenue.csv", index=False)

    # Save Parquet (optional; requires pyarrow)
    try:
        fact_orders.to_parquet(f"{out_dir}/fact_orders.parquet", index=False)
    except Exception as exc:
        print("Parquet export skipped:", exc)


if __name__ == "__main__":
    main()


