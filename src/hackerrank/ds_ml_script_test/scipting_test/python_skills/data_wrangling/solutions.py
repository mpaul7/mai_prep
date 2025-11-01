"""
Hackerrank-style pandas solutions with a simple CLI.

Usage examples:
  python solutions.py q1 --input customers.csv --output cleaned_customers.csv
  python solutions.py q2 --input orders.csv --output monthly_revenue.csv
  python solutions.py q3 --orders orders.csv --customers customers.csv --top_n 5 --output top_customers.csv
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd


def _clean_text_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.lower()
    return s


def _parse_currency(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.replace(r"[^0-9.]+", "", regex=True)
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors="coerce")


def solve_q1(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    name = _clean_text_series(df.get("name"))
    city = _clean_text_series(df.get("city")).fillna("")
    state = _clean_text_series(df.get("state")).fillna("")
    phone_digits = df.get("phone").astype("string").str.replace(r"\D+", "", regex=True)
    phone_digits = phone_digits.fillna("")

    out = pd.DataFrame(
        {
            "name_clean": name,
            "city_clean": city,
            "state_clean": state,
            "phone_digits": phone_digits,
        }
    )
    out.to_csv(output_path, index=False)


def solve_q2(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    df["price_num"] = _parse_currency(df["price"])  # may be NaN
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["price_num", "order_date"]).copy()

    median_qty = int(pd.Series(df["quantity"]).dropna().median()) if df["quantity"].notna().any() else 1
    df["quantity_filled"] = pd.Series(df["quantity"]).fillna(median_qty).astype(int)
    df["revenue"] = df["price_num"] * df["quantity_filled"]
    df.to_csv("./results/q2_df.csv", index=False)

    monthly = (
        df.assign(year=df["order_date"].dt.year, month=df["order_date"].dt.month)
        .groupby(["year", "month"], as_index=False)
        .agg(monthly_revenue=("revenue", "sum"))
        .sort_values(["year", "month"]) 
    )
    monthly.to_csv(output_path, index=False)


def solve_q3(orders_path: str, customers_path: str, output_path: str, top_n: int) -> None:
    orders = pd.read_csv(orders_path)
    customers = pd.read_csv(customers_path)

    orders["price_num"] = _parse_currency(orders["price"])  # may be NaN
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    orders = orders.dropna(subset=["price_num", "order_date"]).copy()

    median_qty = int(pd.Series(orders["quantity"]).dropna().median()) if orders["quantity"].notna().any() else 1
    orders["quantity_filled"] = pd.Series(orders["quantity"]).fillna(median_qty).astype(int)
    orders["revenue"] = orders["price_num"] * orders["quantity_filled"]

    agg = (
        orders.groupby("customer_id", as_index=False)
        .agg(total_revenue=("revenue", "sum"), order_count=("order_id", "nunique"))
    )
    print(agg)

    out = (
        agg.merge(customers[["customer_id", "name"]], how="left", on="customer_id")
        .sort_values(["total_revenue", "customer_id"], ascending=[False, True])
        .head(top_n)
        .loc[:, ["customer_id", "name", "total_revenue", "order_count"]]
    )
    out.to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hackerrank-style pandas solutions")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("q1", help="Clean Customer Directory")
    p1.add_argument("--input", required=True, help="Input CSV path")
    p1.add_argument("--output", required=True, help="Output CSV path")

    p2 = sub.add_parser("q2", help="Monthly Revenue from Orders")
    p2.add_argument("--input", required=True, help="Input CSV path")
    p2.add_argument("--output", required=True, help="Output CSV path")

    p3 = sub.add_parser("q3", help="Top-N Customers by Revenue")
    p3.add_argument("--orders", required=True, help="Orders CSV path")
    p3.add_argument("--customers", required=True, help="Customers CSV path")
    p3.add_argument("--top_n", type=int, required=True, help="Top N customers to return")
    p3.add_argument("--output", required=True, help="Output CSV path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "q1":
        """
        python3 /home/mpaul/projects/mpaul/mai_prep/src/ds_ml_script_test/scipting_test/python_skills/data_wrangling/solutions.py q1 --input /home/mpaul/projects/mpaul/mai_prep/data/customers_raw.csv  --output /home/mpaul/projects/mpaul/mai_prep/results/solution_groupy/q1.csv
        """
        solve_q1(args.input, args.output)
    elif args.command == "q2":
        """
        python3 /home/mpaul/projects/mpaul/mai_prep/src/ds_ml_script_test/scipting_test/python_skills/data_wrangling/solutions.py q2 --input /home/mpaul/projects/mpaul/mai_prep/data/orders_raw.csv  --output /home/mpaul/projects/mpaul/mai_prep/results/solution_groupy/q2.csv
        """
        solve_q2(args.input, args.output)
    elif args.command == "q3":
        """
        python3 /home/mpaul/projects/mpaul/mai_prep/src/ds_ml_script_test/scipting_test/python_skills/data_wrangling/solutions.py q3 --orders /home/mpaul/projects/mpaul/mai_prep/data/orders_raw.csv --customers /home/mpaul/projects/mpaul/mai_prep/data/customers_raw.csv --top_n 5  --output /home/mpaul/projects/mpaul/mai_prep/results/solution_groupy/q3.csv
        
        """
        solve_q3(args.orders, args.customers, args.output, args.top_n)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()



