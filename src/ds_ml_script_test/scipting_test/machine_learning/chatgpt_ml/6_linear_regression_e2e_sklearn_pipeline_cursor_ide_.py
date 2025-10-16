"""
End-to-end Linear Regression (HackerRank-style):

Steps:
- Load dataset (CSV with numeric/categorical features and target `y`)
- Split train/test (80/20)
- Preprocess:
  - Impute missing values (median for numeric, most frequent for categorical)
  - One-hot encode categoricals
  - Scale numeric features
  - Drop highly correlated numeric features (> 0.95 absolute)
- Train LinearRegression
- Predict on test
- Evaluate: MAE, MSE, RMSE, R^2

Usage:
  python 6_linear_regression_e2e_cursor_ide.py --train train.csv [--target y] [--test test.csv]

If --test is not provided, the test split is taken from --train.
Outputs:
- Prints metrics
- Saves metrics to ./results/lr_e2e_metrics.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def remove_high_correlation(X_num: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """Return list of numeric columns to drop based on correlation threshold."""
    if X_num.shape[1] <= 1:
        return []
    corr = X_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def train_and_evaluate(train_df: pd.DataFrame, target: str, test_df: pd.DataFrame | None = None) -> dict:
    if test_df is None:
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=123)

    X_train, y_train = split_features_target(train_df, target)
    X_test, y_test = split_features_target(test_df, target)

    numeric_cols, categorical_cols = identify_feature_types(X_train)

    # Correlation-based removal on numeric features only (fit on training)
    numeric_to_drop = remove_high_correlation(X_train[numeric_cols])
    numeric_cols_pruned = [c for c in numeric_cols if c not in numeric_to_drop]

    preprocessor = build_preprocessor(numeric_cols_pruned, categorical_cols)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
        "dropped_numeric_due_to_corr": ",".join(numeric_to_drop) if numeric_to_drop else "",
        "n_numeric_used": len(numeric_cols_pruned),
        "n_categorical_used": len(categorical_cols),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E Linear Regression with preprocessing")
    parser.add_argument("--train", required=True, help="Path to training CSV containing target column")
    parser.add_argument("--target", default="y", help="Target column name")
    parser.add_argument("--test", default=None, help="Optional test CSV path with same columns as train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train)

    if args.test is not None and os.path.exists(args.test):
        test_df = pd.read_csv(args.test)
    else:
        test_df = None

    metrics = train_and_evaluate(train_df, target=args.target, test_df=test_df)
    print("Metrics:", metrics)

    os.makedirs("./results", exist_ok=True)
    pd.DataFrame([metrics]).to_csv("./results/lr_e2e_metrics.csv", index=False)


if __name__ == "__main__":
    main()


