import numpy as np


"""
Perform simple linear regression on a single feature (x) and target (y)
without using sklearn or pandas.
""" 
# Example
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]


# Convert to numpy arrays
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

# Handle missing values (replace with mean)
if np.isnan(x).any():
    x[np.isnan(x)] = np.nanmean(x)
if np.isnan(y).any():
    y[np.isnan(y)] = np.nanmean(y)

# Normalize x for stability
x = (x - np.mean(x)) / np.std(x)

# Calculate slope and intercept
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

if denominator == 0:
    raise ValueError("Cannot perform regression: no variation in x variable")

slope = numerator / denominator
intercept = y_mean - slope * x_mean

# Predictions
y_pred = slope * x + intercept

# Metrics
mse = np.mean((y - y_pred) ** 2)
mae = np.mean(np.abs(y - y_pred))
r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2))

print(f"\nModel Performance:")
print(f"--------------------------------")
print(f"slope: {slope}")
print(f"intercept: {intercept}")
print(f"mse: {mse}")
print(f"--------------------------------")
print(f"mae: {mae}")
print(f"r2: {r2}")




