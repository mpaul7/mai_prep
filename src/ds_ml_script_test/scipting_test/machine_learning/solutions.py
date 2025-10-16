"""
Hackerrank-style ML solutions (Linear Regression) with a simple CLI.

Usage examples:
  python solutions.py q1 --train train.csv --output metrics.csv
  python solutions.py q2 --train train.csv --output cv.csv
  python solutions.py q3 --train train.csv --holdout holdout.csv --output preds.csv
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


def _split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = df["y"].values
    X = df.drop(columns=["y"]).values
    return X, y


def solve_q1(train_path: str, output_path: str) -> None:
    df = pd.read_csv(train_path)
    X, y = _split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    out = pd.DataFrame([[mae, mse, rmse, r2]], columns=["mae", "mse", "rmse", "r2"])
    out.to_csv(output_path, index=False)


def solve_q2(train_path: str, output_path: str) -> None:
    df = pd.read_csv(train_path)
    X, y = _split_xy(df)
    model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    scores = cross_val_score(model, X, y, scoring="r2", cv=cv)
    out = pd.DataFrame([[float(np.mean(scores)), float(np.std(scores))]], columns=["cv_r2_mean", "cv_r2_std"])
    out.to_csv(output_path, index=False)


def solve_q3(train_path: str, holdout_path: str, output_path: str) -> None:
    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)
    X, y = _split_xy(train_df)
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(holdout_df.values)
    pd.DataFrame({"y_hat": y_hat}).to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hackerrank-style ML LR solutions")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("q1", help="Fit Linear Regression and Report Test Metrics")
    p1.add_argument("--train", required=True)
    p1.add_argument("--output", required=True)

    p2 = sub.add_parser("q2", help="Cross-validated R^2")
    p2.add_argument("--train", required=True)
    p2.add_argument("--output", required=True)

    p3 = sub.add_parser("q3", help="Predict on Holdout")
    p3.add_argument("--train", required=True)
    p3.add_argument("--holdout", required=True)
    p3.add_argument("--output", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "q1":
        solve_q1(args.train, args.output)
    elif args.command == "q2":
        solve_q2(args.train, args.output)
    elif args.command == "q3":
        solve_q3(args.train, args.holdout, args.output)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()




