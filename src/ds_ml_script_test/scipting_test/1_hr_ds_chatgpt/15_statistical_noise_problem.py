"""_summary_
Concepts Covered
===============

Statistical Noise (ε)
====================
- Random variation in data
- Modeled as ε ~ N(0, σ²) 
- Added to true relationship

Linear Regression
=================
- Fits Y = mX + c + ε
- Uses least squares estimation
- Minimizes squared errors

Error Metrics
=============
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error  
- MASE: Mean Absolute Scaled Error

HackerRank Input Format
=======================
Line 1: n (number of points)
Line 2: X values (space-separated)
Line 3: m c sigma (model parameters)

Definition:

MASE measures the accuracy of a model relative to a simple baseline model — usually the naïve forecast.

It tells you:

“How much better (or worse) is my model compared to just predicting the last value?”

🧠 Formula
MASE = MAE_model / MAE_naive

For a dataset of n observations:

MASE = MAE_model / MAE_naive

MASE = 1/n * Σ|y_t - y_t| / 1/n-1 * Σ|y_t - y_t-1| = MAE_model / MAE_naive




Where:
y_t: actual observed value
haty_t: predicted value from your model
[y_t - y_t-1: one step difference between actual observed value and predicted value

"""

import numpy as np

# ------------------------------
# Input (HackerRank-style)
# ------------------------------
# Example Input:
n = "6"
X_values = "0 1 2 3 4 5"
m_c_sigma = "2 1 1"
# (n, X values, m, c, sigma)
n = int(n.strip())
X = np.array(list(map(float, X_values.split())))
print(f"X: {X}")
m, c, sigma = map(float, m_c_sigma.split())

# ------------------------------
# Step 1: Generate noisy data
# ------------------------------
np.random.seed(42)  # for reproducibility
noise = np.random.normal(0, sigma, size=n)
print(f"noise: {noise}")
Y_true = m * X + c  
print(f"Y_true: {Y_true}")

Y_noisy = Y_true + noise
print(f"Y_noisy: {Y_noisy}")
# ------------------------------
# Step 2: Fit Linear Regression using Least Squares
# ------------------------------
X_mean = np.mean(X)
Y_mean = np.mean(Y_noisy)

# slope (m_hat) and intercept (c_hat)
numerator = np.sum((X - X_mean) * (Y_noisy - Y_mean))
denominator = np.sum((X - X_mean) ** 2)
m_hat = numerator / denominator
c_hat = Y_mean - m_hat * X_mean

# predictions
Y_pred = m_hat * X + c_hat

# ------------------------------
# Step 3: Calculate Error Metrics
# ------------------------------
errors = Y_noisy - Y_pred
mse = np.mean(errors ** 2)
rmse = np.sqrt(mse)

# MASE: Mean Absolute Scaled Error
print(f'Y_noisy: {Y_noisy}')
print(f'Y_pred: {Y_pred}')
print(f'np.diff(Y_noisy): {np.diff(Y_noisy)}')
naive_forecast = np.abs(np.diff(Y_noisy))
if np.mean(naive_forecast) == 0:
    mase = 0.0
else:
    mase = np.mean(np.abs(errors)) / np.mean(naive_forecast)

# ------------------------------
# Step 4: Print Results
# ------------------------------
print(f"Slope (m_hat): {m_hat:.2f}")
print(f"Intercept (c_hat): {c_hat:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MASE: {mase:.2f}")
