import numpy as np

# ---------------------------
# 1. Input
# ---------------------------
X_values = "1 2 3 4 5"
Y_values = "2 4 5 4 5"
x = list(map(float, X_values.split()))       # Independent variable
y_true = list(map(float, Y_values.split()))  # Actual dependent variable

x = np.array(x)
y_true = np.array(y_true)
print(f"x: {x}")
print(f"y_true: {y_true}")
# ---------------------------
# 2. Linear Regression (Simple)
# ---------------------------
x_mean = np.mean(x)
print(f"x_mean: {x_mean}")
y_mean = np.mean(y_true)
print(f"y_mean: {y_mean}")
# Slope (m)
m = np.sum((x - x_mean) * (y_true - y_mean)) / np.sum((x - x_mean)**2)
print(f"m: {m}")
# Intercept (c)
c = y_mean - m * x_mean
print(f"c: {c}")
# Predicted Y values
y_pred = m * x + c
print(f"y_pred: {y_pred}")
# ---------------------------
# 3. Error Metrics
# ---------------------------

# Mean Squared Error (MSE)
mse_val = np.mean((y_true - y_pred)**2)
print(f"mse_val: {mse_val}")
# Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mse_val)
print(f"rmse_val: {rmse_val}")
# Mean Absolute Scaled Error (MASE)
n = len(y_true)
if n < 2:
    mase_val = float('nan')  # Not enough data
else:
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    mase_val = mae_model / mae_naive
print(f"mase_val: {mase_val}")
# ---------------------------
# 4. Output
# ---------------------------
print(f"Regression Line: Y = {round(m,2)}X + {round(c,2)}")
print(f"Predicted Y values: {np.round(y_pred,2).tolist()}")
print(f"MSE: {round(mse_val,2)}")
print(f"RMSE: {round(rmse_val,2)}")
print(f"MASE: {round(mase_val,2)}")
