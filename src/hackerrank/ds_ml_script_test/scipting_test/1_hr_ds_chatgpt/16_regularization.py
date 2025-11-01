"""_summary_
Definition:

Regularization is a set of techniques used to reduce overfitting by penalizing large coefficients in machine learning models (especially regression).

The goal is to improve generalization — the model’s ability to perform well on unseen data.

Why We Need It
=============

Without regularization:
----------------------
- The model fits both the signal and noise in data → overfitting
- Predictions on new data become unstable or inaccurate

With regularization:
-------------------
- The model focuses on the main trends
- Coefficients are smaller and more stable


Conceptual Multiple Choice Questions (Expected on HackerRank)
----------------------------------------------------------

Q1. What is the main purpose of regularization?
• To prevent overfitting ✓

Q2. What happens to coefficients as λ increases in Ridge regression? 
• They shrink towards zero ✓

Q3. Which regularization can eliminate some features completely?
• Lasso (L1) ✓

Q4. Which regularization keeps all features but reduces their magnitude?
• Ridge (L2) ✓
"""
# ------------------------------
# Linear Regression with L2 Regularization (Ridge)
# ------------------------------
import numpy as np

# ------------------------------
# Hardcoded Input
# ------------------------------
# Dataset: small linear relation with slight noise
# y = 1 + 2*x1 + 0.5*x2 + noise
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
], dtype=float)

y = np.array([6.1, 8.9, 11.2, 13.8, 16.4])  # noisy target
lmbd = 0.5  # regularization strength λ

# ------------------------------
# Add bias (intercept) column
# ------------------------------
n = X.shape[0]
X = np.c_[np.ones(n), X]

# ------------------------------
# Ridge Regression Formula:
# β = (XᵀX + λI)⁻¹ Xᵀy
# ------------------------------
I = np.eye(X.shape[1])
I[0, 0] = 0  # don't regularize bias

beta = np.linalg.inv(X.T @ X + lmbd * I) @ X.T @ y
y_pred = X @ beta

# ------------------------------
# Error Metrics
# ------------------------------
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)

# ------------------------------
# Output
# ------------------------------
print("==== Ridge Regression (L2 Regularization) ====")
print("Lambda (λ):", lmbd)
print("MSE:", mse)