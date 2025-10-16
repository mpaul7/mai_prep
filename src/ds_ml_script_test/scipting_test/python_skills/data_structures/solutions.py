"""
Hackerrank-style data structures solutions with a simple CLI.

Usage examples:
  python solutions.py q1 --input words.txt --output groups.txt
  python solutions.py q2 --inputs d1.json d2.json d3.json --output merged.json
  python solutions.py q3 --input items.txt --k 3 --output topk.csv
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import List


def solve_q1(input_path: str, output_path: str) -> None:
    # Read one word per line
    with open(input_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    groups_dict = defaultdict(list)
    for w in words:
        signature = "".join(sorted(w))
        groups_dict[signature].append(w)

    groups = [sorted(g) for g in groups_dict.values()]
    groups.sort(key=lambda g: (-len(g), g[0]))

    with open(output_path, "w", encoding="utf-8") as f:
        for g in groups:
            f.write(" ".join(g) + "\n")


def solve_q2(inputs: List[str], output_path: str) -> None:
    merged = Counter()
    for path in inputs:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        merged.update(d)
    # Sort keys ascending
    out = {k: int(merged[k]) for k in sorted(merged)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))


def solve_q3(input_path: str, k: int, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]
    counts = Counter(items)
    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("item,frequency\n")
        for item, freq in top:
            f.write(f"{item},{freq}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hackerrank-style data structures solutions")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("q1", help="Anagram Groups")
    p1.add_argument("--input", required=True)
    p1.add_argument("--output", required=True)

    p2 = sub.add_parser("q2", help="Merge Dictionaries with Sum")
    p2.add_argument("--inputs", nargs="+", required=True)
    p2.add_argument("--output", required=True)

    p3 = sub.add_parser("q3", help="K Most Frequent Items with Tie-Breaks")
    p3.add_argument("--input", required=True)
    p3.add_argument("--k", type=int, required=True)
    p3.add_argument("--output", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "q1":
        solve_q1(args.input, args.output)
    elif args.command == "q2":
        solve_q2(args.inputs, args.output)
    elif args.command == "q3":
        solve_q3(args.input, args.k, args.output)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
