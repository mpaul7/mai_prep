# ===========================================
# Regularization Practice:
# Ridge vs Lasso vs ElasticNet
# ===========================================

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# --------------------------
# 1. Generate synthetic data
# --------------------------
X, y = make_regression(
    n_samples=200,       # number of samples
    n_features=10,       # total features
    n_informative=5,     # relevant features
    noise=20,            # add some noise
    random_state=42
)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 2. Train Ridge Regression (L2)
# --------------------------
ridge = Ridge(alpha=1.0)  # alpha = λ = regularization strength
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# --------------------------
# 3. Train Lasso Regression (L1)
# --------------------------
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# --------------------------
# 4. Train ElasticNet (L1 + L2)
# --------------------------
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)

# --------------------------
# 5. Evaluate all models
# --------------------------
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
elastic_mse = mean_squared_error(y_test, y_pred_elastic)

# Combine results into a DataFrame for comparison
results = pd.DataFrame({
    'Model': ['Ridge', 'Lasso', 'ElasticNet'],
    'MSE': [ridge_mse, lasso_mse, elastic_mse],
    'Num_NonZero_Coeff': [
        np.sum(ridge.coef_ != 0),
        np.sum(lasso.coef_ != 0),
        np.sum(elastic.coef_ != 0)
    ]
})

print("==== Regularization Model Comparison ====")
print(results.to_string(index=False))

# --------------------------
# 6. Insights
# --------------------------
print("\nRidge shrinks all coefficients but keeps them non-zero.")
print("Lasso can zero-out some coefficients (feature selection).")
print("ElasticNet combines both effects, balancing sparsity and stability.")
