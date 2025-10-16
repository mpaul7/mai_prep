"""
Linear Regression Example: training and evaluation with scikit-learn.

Demonstrates:
- Train/test split
- Feature scaling (optional)
- LinearRegression fit
- Metrics: MAE, MSE, RMSE, R^2
- Cross-validation (KFold)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


def make_synthetic(n_samples: int = 500, n_features: int = 8, noise: float = 10.0, random_state: int = 42) -> pd.DataFrame:
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        noise=noise,
        coef=True,
        random_state=random_state,
    )
    columns = [f"x{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["y"] = y
    return df


def train_and_evaluate(df: pd.DataFrame, scale_features: bool = False) -> dict:
    X = df.drop(columns=["y"]).values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Cross-validated R^2
    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    cv_scores = cross_val_score(model, X, y, scoring="r2", cv=cv)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
    }


def main() -> None:
    df = make_synthetic()
    print(df)
    metrics = train_and_evaluate(df, scale_features=False)
    print("Metrics:", metrics)
    pd.DataFrame([metrics]).to_csv("./results/lr_metrics.csv", index=False)


if __name__ == "__main__":
    main()


