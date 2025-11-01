"""
Hackerrank-style statistics solutions with a simple CLI.

Usage examples:
  python solutions.py q1 --input xy.csv
  python solutions.py q2 --n 10 --k 4 --p 0.3
  python solutions.py q3 --input data.csv --target A
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats


def solve_q1(input_path: str) -> None:
    df = pd.read_csv(input_path)
    df = df[["x", "y"]].dropna()
    if len(df) == 0:
        print("nan")
        return
    r = df["x"].corr(df["y"], method="pearson")
    print(f"{r:.3f}")


def solve_q2(n: int, k: int, p: float) -> None:
    prob = stats.binom.sf(k - 1, n=n, p=p)
    print(f"{prob:.5f}")


def solve_q3(input_path: str, target: str) -> None:
    df = pd.read_csv(input_path)
    # Normalize flag to boolean
    if df["flag"].dtype != bool:
        df["flag"] = df["flag"].astype(int) == 1
    sub = df[df["flag"] == True]
    if len(sub) == 0:
        print("0.000")
        return
    p = (sub["group"] == target).mean()
    print(f"{p:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hackerrank-style statistics solutions")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("q1", help="Pearson Correlation")
    p1.add_argument("--input", required=True)

    p2 = sub.add_parser("q2", help="Binomial Tail Probability")
    p2.add_argument("--n", type=int, required=True)
    p2.add_argument("--k", type=int, required=True)
    p2.add_argument("--p", type=float, required=True)

    p3 = sub.add_parser("q3", help="Conditional Probability from Contingency")
    p3.add_argument("--input", required=True)
    p3.add_argument("--target", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "q1":
        solve_q1(args.input)
    elif args.command == "q2":
        solve_q2(args.n, args.k, args.p)
    elif args.command == "q3":
        solve_q3(args.input, args.target)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()


