"""
Reusable data-processing functions examples.

Demonstrates:
- Pure, testable functions with clear signatures and docstrings
- Type hints and input validation
- Composability and small utilities reused across tasks
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def coalesce(strings: Sequence[str | None], fallback: str = "") -> List[str]:
    """Return a list where None/empty entries become `fallback`.

    Empty after strip is considered missing.
    """
    out: List[str] = []
    for s in strings:
        if s is None:
            out.append(fallback)
        else:
            val = s.strip()
            out.append(val if val else fallback)
    return out


def normalize_whitespace(strings: Sequence[str]) -> List[str]:
    """Trim, collapse spaces, and lowercase."""
    return [" ".join(s.strip().split()).lower() for s in strings]


def robust_to_datetime(series: pd.Series) -> pd.Series:
    """Parse to datetime with coercion; returns a datetime64 series with NaT for invalids."""
    return pd.to_datetime(series, errors="coerce")


def safe_median(series: pd.Series) -> float:
    """Median ignoring NaN; returns 0.0 if all missing."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return float(s.median())
    return 0.0


def zscore(series: pd.Series) -> pd.Series:
    """Standard score with NaN-safe behavior."""
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mean) / std


def aggregate_by(df: pd.DataFrame, group_cols: Sequence[str], agg_map: dict) -> pd.DataFrame:
    """Convenience wrapper around groupby-agg returning a flat DataFrame."""
    return df.groupby(list(group_cols), as_index=False).agg(**agg_map)


def run_example() -> None:
    names = ["  Alice  ", None, "bob", "", "  CAROL  "]
    print("coalesce:", coalesce(names, fallback="unknown"))
    print("normalize:", normalize_whitespace([" New   York ", "san  francisco"]))

    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "value": [1, np.nan, 3, 4, 5],
            "date": ["2023-01-01", "bad", "2023-01-03", "2023-01-04", "2023-01-05"],
        }
    )
    print("robust_to_datetime:\n", robust_to_datetime(df["date"]))
    print("safe_median:", safe_median(df["value"]))
    print("zscore:\n", zscore(df["value"]))

    agg = aggregate_by(
        df.assign(value=df["value"].fillna(safe_median(df["value"]))),
        group_cols=["group"],
        agg_map={"sum_value": ("value", "sum"), "mean_value": ("value", "mean")},
    )
    print("aggregate_by:\n", agg)


if __name__ == "__main__":
    run_example()


