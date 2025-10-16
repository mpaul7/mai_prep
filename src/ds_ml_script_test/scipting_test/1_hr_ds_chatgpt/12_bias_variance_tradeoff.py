"""_summary_
Concept

Bias-Variance Tradeoff:

1. Bias
   - Error due to overly simple assumptions in the model (underfitting)
   - High bias means model fails to capture patterns in data

2. Variance  
   - Error due to sensitivity to fluctuations in training data (overfitting)
   - High variance means model fits training data too closely, fails on new data

3. Total Expected Error
   - Formula: Bias² + Variance + Irreducible Error
   - Goal: Minimize total error by balancing bias and variance
"""

import numpy as np

# ---------------------------
# 1. True function
# ---------------------------
def true_function(x):
    return 2*x + 3

# ---------------------------
# 2. Simulate datasets
# ---------------------------
np.random.seed(42)
n_samples = 50
x = np.linspace(0, 10, n_samples)
print(f"x: {x}")
# Generate multiple datasets with noise
datasets = [true_function(x) + np.random.normal(0, 2, n_samples) for _ in range(100)]
print(f"datasets: {datasets}")
# ---------------------------
# 3. Fit models & compute predictions
# ---------------------------

# Low complexity: linear regression (Y = mX + c)
y_preds_linear = []
for y in datasets:
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
    c = y_mean - m*x_mean
    y_pred = m*x + c
    y_preds_linear.append(y_pred)

y_preds_linear = np.array(y_preds_linear)

# ---------------------------
# 4. Compute bias and variance
# ---------------------------
# Bias: squared difference between average prediction and true function
y_true = true_function(x)
avg_pred = np.mean(y_preds_linear, axis=0)
bias_squared = np.mean((avg_pred - y_true)**2)

# Variance: average of variance across datasets
variance = np.mean(np.var(y_preds_linear, axis=0))

print(f"Bias^2 (Linear Regression): {round(bias_squared,2)}")
print(f"Variance (Linear Regression): {round(variance,2)}")
