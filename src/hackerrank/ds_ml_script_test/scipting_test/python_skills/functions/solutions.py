"""
Hackerrank-style functions solutions with a simple CLI.

Usage examples:
  python solutions.py q1 --input values.txt --fallback unknown --output out.txt
  python solutions.py q2 --input text.txt --output normalized.txt
  python solutions.py q3 --input data.csv --output agg.csv
"""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd


def coalesce_str(values: List[str | None], fallback: str) -> List[str]:
    out: List[str] = []
    for s in values:
        if s is None:
            out.append(fallback)
        else:
            val = s.strip()
            out.append(val if val else fallback)
    return out


def normalize_text(values: List[str]) -> List[str]:
    return [" ".join(s.strip().split()).lower() for s in values]


def aggregate_by(df: pd.DataFrame, group_cols: list[str], agg_map: dict) -> pd.DataFrame:
    return df.groupby(group_cols, as_index=False).agg(**agg_map)


def solve_q1(input_path: str, fallback: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        values = [line.rstrip("\n") if line.strip() != "" else "" for line in f]
    # Treat empty lines as empty strings; None is not present in files, but requirement retained
    out = coalesce_str(values, fallback)
    with open(output_path, "w", encoding="utf-8") as f:
        for v in out:
            f.write(v + "\n")


def solve_q2(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        values = [line.rstrip("\n") for line in f]
    out = normalize_text(values)
    with open(output_path, "w", encoding="utf-8") as f:
        for v in out:
            f.write(v + "\n")


def solve_q3(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    agg = aggregate_by(df, group_cols=["group"], agg_map={"sum_value": ("value", "sum"), "mean_value": ("value", "mean")})
    agg.to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hackerrank-style functions solutions")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("q1", help="coalesce_str")
    p1.add_argument("--input", required=True)
    p1.add_argument("--fallback", required=True)
    p1.add_argument("--output", required=True)

    p2 = sub.add_parser("q2", help="normalize_text")
    p2.add_argument("--input", required=True)
    p2.add_argument("--output", required=True)

    p3 = sub.add_parser("q3", help="aggregate_by")
    p3.add_argument("--input", required=True)
    p3.add_argument("--output", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "q1":
        solve_q1(args.input, args.fallback, args.output)
    elif args.command == "q2":
        solve_q2(args.input, args.output)
    elif args.command == "q3":
        solve_q3(args.input, args.output)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
