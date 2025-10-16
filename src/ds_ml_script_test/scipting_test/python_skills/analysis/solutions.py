"""
Hackerrank-style analysis solutions with a simple CLI.

Usage examples:
  python solutions.py q1 --input data.csv --output contribution.csv
  python solutions.py q2 --input weekly.csv --output rolling.csv
  python solutions.py q3 --input data.csv --top_n 2 --output topn.csv
"""

from __future__ import annotations

import argparse
import pandas as pd


def solve_q1(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    grouped = df.groupby(["region", "category"], as_index=False).agg(revenue_sum=("revenue", "sum"))
    total_by_region = grouped.groupby("region", as_index=False)["revenue_sum"].sum().rename(columns={"revenue_sum": "region_total"})
    out = (
        grouped.merge(total_by_region, on="region", how="left")
        .assign(share_pct=lambda d: (d["revenue_sum"] / d["region_total"]) * 100)
        .drop(columns=["region_total"]) 
        .sort_values(["region", "share_pct"], ascending=[True, False])
    )
    out["share_pct"] = out["share_pct"].round(2)
    out.to_csv(output_path, index=False)


def solve_q2(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    weekly = df.groupby(["region", "week"], as_index=False).agg(weekly_revenue=("revenue", "sum"))
    weekly = weekly.sort_values(["region", "week"])  # ensure ordering for rolling
    weekly["rolling_4w"] = (
        weekly.groupby("region")["weekly_revenue"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    )
    weekly.to_csv(output_path, index=False)


def solve_q3(input_path: str, output_path: str, top_n: int) -> None:
    df = pd.read_csv(input_path)
    agg = df.groupby(["region", "category"], as_index=False).agg(revenue_sum=("revenue", "sum"))
    agg["rank"] = agg.groupby("region")["revenue_sum"].rank(ascending=False, method="first")
    out = agg[agg["rank"] <= top_n].sort_values(["region", "rank", "category"]).reset_index(drop=True)
    out.to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hackerrank-style analysis solutions")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("q1", help="Category Contribution by Region")
    p1.add_argument("--input", required=True)
    p1.add_argument("--output", required=True)

    p2 = sub.add_parser("q2", help="Rolling Weekly Revenue")
    p2.add_argument("--input", required=True)
    p2.add_argument("--output", required=True)

    p3 = sub.add_parser("q3", help="Top-N Categories by Region")
    p3.add_argument("--input", required=True)
    p3.add_argument("--top_n", type=int, required=True)
    p3.add_argument("--output", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "q1":
        solve_q1(args.input, args.output)
    elif args.command == "q2":
        solve_q2(args.input, args.output)
    elif args.command == "q3":
        solve_q3(args.input, args.output, args.top_n)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
