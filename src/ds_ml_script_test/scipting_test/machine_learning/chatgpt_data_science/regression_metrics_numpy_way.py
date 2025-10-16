import numpy as np

# Example true and predicted values
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

# MAE
mae = np.mean(np.abs(y_true - y_pred))

# MSE
mse = np.mean((y_true - y_pred)**2)

# RMSE
rmse = np.sqrt(mse)

# R² Score
ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
r2_score = 1 - ss_res / ss_tot

print("Regression Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2_score)
