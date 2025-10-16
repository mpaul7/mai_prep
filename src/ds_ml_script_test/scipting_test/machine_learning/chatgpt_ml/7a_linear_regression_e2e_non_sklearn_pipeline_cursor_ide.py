"""
Single-function end-to-end Linear Regression without scikit-learn.

- Generates synthetic data when no files provided
- Cleans categorical text, imputes numeric/categorical
- One-hot encodes categoricals, scales numerics
- Trains via normal equation (with small ridge term)
- Evaluates MAE, MSE, RMSE, R^2

Usage:
  python 7a_linear_regression_e2e_non_sklearn_pipeline_cursor_ide.py [--train train.csv] [--test test.csv] [--target y] [--test_size 0.2] [--random_state 123]
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def solve_end_to_end_single(train_path: str | None, test_path: str | None, target: str, test_size: float, random_state: int) -> dict:
    # 1) Load or synthesize training data
    # -------------------------------------

    rng = np.random.default_rng(random_state)
    n_samples, n_numeric, n_categorical = 600, 6, 2
    X_num = rng.normal(0.0, 1.0, size=(n_samples, n_numeric))
    num_cols = [f"x{i}" for i in range(n_numeric)]
    df_full = pd.DataFrame(X_num, columns=num_cols)
    cat_cols = [f"cat{i}" for i in range(n_categorical)]
    categories = [np.array(["a", "b", "c"]) for _ in range(n_categorical)]
    for i, col in enumerate(cat_cols):
        df_full[col] = rng.choice(categories[i], size=n_samples, replace=True)
    # true relation
    w_num = rng.normal(2.0, 1.0, size=n_numeric)
    y = df_full[num_cols].values @ w_num
    for i, col in enumerate(cat_cols):
        effects = {cat: float(rng.normal(0.0, 3.0)) for cat in categories[i]}
        y += df_full[col].map(effects).astype(float).values
    y += rng.normal(0.0, 2.0, size=n_samples)
    # missingness
    for col in num_cols:
        mask = rng.random(n_samples) < 0.08
        df_full.loc[mask, col] = np.nan
    for col in cat_cols:
        mask = rng.random(n_samples) < 0.08
        df_full.loc[mask, col] = None
    df_full[target] = y
    print(f"df_full: {df_full}")

    # Split train/test
    # ----------------
    
    X_full = df_full.drop(columns=[target]).copy()
    y_full = df_full[target].astype(float).values
    
    # # 2) Identify types and normalize categorical text for full data
    # ----------------------------------------------------------------
    numeric_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_full.columns if c not in numeric_cols]
    for col in categorical_cols:
        X_full[col] = X_full[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.lower()

    # 3) Split or load external test
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df_full))
    rng.shuffle(idx)
    cut = int(round(test_size * len(idx)))
    test_idx, train_idx = idx[:cut], idx[cut:]
    train_df = df_full.iloc[train_idx].reset_index(drop=True)
    test_df = df_full.iloc[test_idx].reset_index(drop=True)

    # 4) Preprocess on TRAIN; apply to TEST (impute, scale, one-hot)
    X_train = train_df.drop(columns=[target]).copy()
    y_train = train_df[target].astype(float).values
    X_test = test_df.drop(columns=[target]).copy()
    y_test = test_df[target].astype(float).values

    num_cols_train = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_train = [c for c in X_train.columns if c not in num_cols_train]
    # normalize categorical
    for col in cat_cols_train:
        X_train[col] = X_train[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
        X_test[col] = X_test[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.lower()

    # compute imputation stats on train
    numeric_medians = {}
    numeric_means = {}
    numeric_stds = {}
    for col in num_cols_train:
        col_vals = pd.to_numeric(X_train[col], errors="coerce")
        med = float(col_vals.median()) if col_vals.notna().any() else 0.0
        m = float(col_vals.fillna(med).mean())
        s = float(col_vals.fillna(med).std(ddof=0))
        numeric_medians[col] = med
        numeric_means[col] = m
        numeric_stds[col] = s if s > 0 else 1.0
    categorical_modes = {}
    for col in cat_cols_train:
        mode_val = X_train[col].mode(dropna=True)
        categorical_modes[col] = (mode_val.iloc[0] if len(mode_val) > 0 else "unknown")

    # impute
    for col in num_cols_train:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(numeric_medians[col])
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(numeric_medians[col])
    for col in cat_cols_train:
        X_train[col] = X_train[col].astype("string").fillna(categorical_modes[col])
        X_test[col] = X_test[col].astype("string").fillna(categorical_modes[col])

    # scale numerics
    for col in num_cols_train:
        X_train[col] = (pd.to_numeric(X_train[col], errors="coerce") - numeric_means[col]) / numeric_stds[col]
        X_test[col] = (pd.to_numeric(X_test[col], errors="coerce") - numeric_means[col]) / numeric_stds[col]

    # one-hot encode
    X_train_dum = pd.get_dummies(X_train, columns=cat_cols_train, dummy_na=False)
    train_dummy_cols = [c for c in X_train_dum.columns if c not in num_cols_train]
    X_test_dum = pd.get_dummies(X_test, columns=cat_cols_train, dummy_na=False)
    for c in train_dummy_cols:
        if c not in X_test_dum.columns:
            X_test_dum[c] = 0
    keep_cols = num_cols_train + train_dummy_cols
    X_train_dum = X_train_dum.reindex(columns=keep_cols, fill_value=0)
    X_test_dum = X_test_dum.reindex(columns=keep_cols, fill_value=0)

    # 5) Fit linear regression (normal equation with ridge)
    X_train_mat = np.hstack([np.ones((X_train_dum.shape[0], 1)), X_train_dum.values])
    X_test_mat = np.hstack([np.ones((X_test_dum.shape[0], 1)), X_test_dum.values])
    l2 = 1e-6
    w = np.linalg.pinv(X_train_mat.T @ X_train_mat + l2 * np.eye(X_train_mat.shape[1])) @ X_train_mat.T @ y_train
    y_pred = X_test_mat @ w

    # 6) Metrics
    mae = float(np.mean(np.abs(y_test - y_pred)))
    mse = float(np.mean((y_test - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    os.makedirs("./results", exist_ok=True)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("./results/lr_non_sklearn_preds_7a.csv", index=False)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": float(r2)}


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Single-function Non-sklearn E2E LR")
#     p.add_argument("--train", default=None)
#     p.add_argument("--test", default=None)
#     p.add_argument("--target", default="y")
#     p.add_argument("--test_size", type=float, default=0.2)
#     p.add_argument("--random_state", type=int, default=123)
#     return p.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    metrics = solve_end_to_end_single(None, None, "y", 0.2, 123)
    print("Metrics:", metrics)
    os.makedirs("./results", exist_ok=True)
    pd.DataFrame([metrics]).to_csv("./results/lr_non_sklearn_metrics_7a.csv", index=False)


