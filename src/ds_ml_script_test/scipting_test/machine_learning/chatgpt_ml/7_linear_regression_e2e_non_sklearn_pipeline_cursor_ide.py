"""
End-to-end Linear Regression without scikit-learn.

Features:
- Train/test split (manual)
- Data cleaning: normalize categorical text
- Imputation: numeric median, categorical most frequent
- One-hot encoding for categorical columns
- Feature scaling for numerics (standardization)
- Model training via normal equation with ridge stabilization
- Evaluation: MAE, MSE, RMSE, R^2

Usage:
  python 7_linear_regression_e2e_non_sklearn_pipeline_cursor_ide_ [--train train.csv] [--target y] [--test test.csv] [--test_size 0.2]

Outputs:
- Prints metrics
- Saves metrics to ./results/lr_non_sklearn_metrics.csv
- Saves predictions for test to ./results/lr_non_sklearn_preds.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def generate_synthetic_df(n_samples: int = 500, n_numeric: int = 5, n_categorical: int = 2, random_state: int = 123) -> pd.DataFrame:
    """Create a synthetic regression dataset with numeric and categorical features and missing values."""
    rng = np.random.default_rng(random_state)

    # Numeric features
    X_num = rng.normal(0.0, 1.0, size=(n_samples, n_numeric))
    num_cols = [f"x{i}" for i in range(n_numeric)]
    df = pd.DataFrame(X_num, columns=num_cols)

    # Categorical features
    cat_cols = [f"cat{i}" for i in range(n_categorical)]
    categories = [np.array(["a", "b", "c"]) for _ in range(n_categorical)]
    for i, col in enumerate(cat_cols):
        df[col] = rng.choice(categories[i], size=n_samples, replace=True)

    # True coefficients for numeric
    w_num = rng.normal(2.0, 1.0, size=n_numeric)
    y = df[num_cols].values @ w_num

    # Category effects
    for i, col in enumerate(cat_cols):
        effects = {cat: float(rng.normal(0.0, 3.0)) for cat in categories[i]}
        y += df[col].map(effects).astype(float).values

    # Noise
    y += rng.normal(0.0, 2.5, size=n_samples)

    # Inject missingness
    for col in num_cols:
        mask = rng.random(n_samples) < 0.1
        df.loc[mask, col] = np.nan
    for col in cat_cols:
        mask = rng.random(n_samples) < 0.1
        df.loc[mask, col] = None

    df["y"] = y
    return df


def train_test_split_manual(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 123) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    test_count = int(round(test_size * len(df)))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def normalize_categorical_text(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        out[col] = out[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
    return out


class SimplePreprocessor:
    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.numeric_medians: Dict[str, float] = {}
        self.numeric_means: Dict[str, float] = {}
        self.numeric_stds: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.dummy_columns: List[str] = []

    def fit(self, X: pd.DataFrame) -> None:
        # Impute statistics
        for col in self.numeric_cols:
            col_vals = pd.to_numeric(X[col], errors="coerce")
            self.numeric_medians[col] = float(col_vals.median()) if col_vals.notna().any() else 0.0
            self.numeric_means[col] = float(col_vals.fillna(self.numeric_medians[col]).mean())
            std = float(col_vals.fillna(self.numeric_medians[col]).std(ddof=0))
            self.numeric_stds[col] = std if std > 0 else 1.0

        for col in self.categorical_cols:
            mode_val = X[col].mode(dropna=True)
            self.categorical_modes[col] = (mode_val.iloc[0] if len(mode_val) > 0 else "unknown")

        # Build dummy columns from training
        X_imputed = self._impute(X.copy())
        X_dummies = pd.get_dummies(X_imputed, columns=self.categorical_cols, dummy_na=False)
        # Note: keep numeric + all dummy columns
        self.dummy_columns = [c for c in X_dummies.columns if c not in self.numeric_cols]

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Numeric impute
        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_medians[col])
        # Categorical impute
        for col in self.categorical_cols:
            X[col] = X[col].astype("string").fillna(self.categorical_modes[col])
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Ensure same columns exist
        for col in self.numeric_cols:
            if col not in X:
                X[col] = np.nan
        for col in self.categorical_cols:
            if col not in X:
                X[col] = np.nan

        X = self._impute(X)

        # Standardize numeric
        for col in self.numeric_cols:
            X[col] = (pd.to_numeric(X[col], errors="coerce") - self.numeric_means[col]) / self.numeric_stds[col]

        # One-hot for categoricals based on train categories
        X_dummies = pd.get_dummies(X, columns=self.categorical_cols, dummy_na=False)
        # Align to training dummy columns (add missing as 0)
        for c in self.dummy_columns:
            if c not in X_dummies.columns:
                X_dummies[c] = 0
        # Drop any unexpected dummy columns (categories unseen in train)
        keep_cols = self.numeric_cols + self.dummy_columns
        X_dummies = X_dummies.reindex(columns=keep_cols, fill_value=0)
        return X_dummies


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_linear_regression_normal_eq(X: np.ndarray, y: np.ndarray, l2: float = 1e-6) -> np.ndarray:
    # Normal equation with ridge stabilization: w = (X^T X + l2 I)^-1 X^T y
    n_features = X.shape[1]
    XtX = X.T @ X
    reg = l2 * np.eye(n_features)
    w = np.linalg.pinv(XtX + reg) @ X.T @ y
    return w


def predict_linear_regression(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-sklearn E2E Linear Regression")
    parser.add_argument("--train", default=None, help="Optional path to training CSV with target column; if omitted, synthetic data is generated")
    parser.add_argument("--target", default="y", help="Target column name")
    parser.add_argument("--test", default=None, help="Optional test CSV path with same columns as train")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size if --test not provided")
    parser.add_argument("--random_state", type=int, default=123, help="Random seed for split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load or generate training data
    if args.train and os.path.exists(args.train):
        train_df_full = pd.read_csv(args.train)
    else:
        train_df_full = generate_synthetic_df(random_state=args.random_state)

    # Identify numeric and categorical columns from the full training data (excluding target)
    X_full, y_full = split_features_target(train_df_full, args.target)
    num_cols, cat_cols = identify_feature_types(X_full)
    X_full = normalize_categorical_text(X_full, cat_cols)

    # Build train/test
    if args.test and os.path.exists(args.test):
        test_df_full = pd.read_csv(args.test)
        X_test_full, y_test = split_features_target(test_df_full, args.target)
        X_test_full = normalize_categorical_text(X_test_full, cat_cols)
        train_df = pd.concat([X_full, y_full], axis=1)
        test_df = pd.concat([X_test_full, y_test], axis=1)
    else:
        df_clean = pd.concat([X_full, y_full], axis=1)
        train_df, test_df = train_test_split_manual(df_clean, test_size=args.test_size, random_state=args.random_state)

    X_train, y_train = split_features_target(train_df, args.target)
    X_test, y_test = split_features_target(test_df, args.target)

    # Re-compute types from TRAIN only
    num_cols_train, cat_cols_train = identify_feature_types(X_train)
    X_train = normalize_categorical_text(X_train, cat_cols_train)
    X_test = normalize_categorical_text(X_test, cat_cols_train)

    pre = SimplePreprocessor(num_cols_train, cat_cols_train)
    pre.fit(X_train)
    X_train_proc = pre.transform(X_train)
    X_test_proc = pre.transform(X_test)

    # Design matrices with intercept
    X_train_mat = add_intercept(X_train_proc.values)
    X_test_mat = add_intercept(X_test_proc.values)
    y_train_arr = y_train.values.astype(float)
    y_test_arr = y_test.values.astype(float)

    # Fit and predict
    w = fit_linear_regression_normal_eq(X_train_mat, y_train_arr, l2=1e-6)
    y_pred = predict_linear_regression(X_test_mat, w)

    # Metrics
    met = metrics_regression(y_test_arr, y_pred)
    print("Metrics:", met)

    os.makedirs("./results", exist_ok=True)
    pd.DataFrame([met]).to_csv("./results/lr_non_sklearn_metrics.csv", index=False)
    pd.DataFrame({"y_true": y_test_arr, "y_pred": y_pred}).to_csv("./results/lr_non_sklearn_preds.csv", index=False)


if __name__ == "__main__":
    main()


