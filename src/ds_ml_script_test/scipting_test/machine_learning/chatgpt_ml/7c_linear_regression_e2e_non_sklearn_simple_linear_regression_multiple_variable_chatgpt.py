import numpy as np

"""
Perform multiple linear regression with both numeric and categorical features,
handling missing values, encoding, normalization, and metrics manually.
"""

# Example test
X = [
    [2104, 5, "Urban"],
    [1416, 3, "Rural"],
    [1534, 3, "Urban"],
    [852, 2, "Suburban"],
    [900, 2, "Urban"]
]

y = [460, 232, 315, 178, 190]

# Step 1: Identify column types (categorical vs numeric)
# Assume mixed data: numeric values or strings

print(f"X: \n{X}")
X_processed = []
for col in zip(*X):
    print(f"col: \n{col}")
    # Check if column is numeric
    if all(isinstance(v, (int, float, np.number)) or str(v).replace('.', '', 1).isdigit() for v in col):
        # Convert to float, replace missing with mean
        col = np.array([float(v) if str(v).replace('.', '', 1).isdigit() else np.nan for v in col])
        col[np.isnan(col)] = np.nanmean(col)
        # Normalize numeric column
        col = (col - np.mean(col)) / np.std(col)
        X_processed.append(col)
    else:
        # Handle categorical column via one-hot encoding
        unique_vals = sorted(list(set(col)))
        one_hot_encoded = np.zeros((len(col), len(unique_vals)))
        for i, val in enumerate(col):
            one_hot_encoded[i, unique_vals.index(val)] = 1
        # Add each one-hot encoded column separately
        for j in range(one_hot_encoded.shape[1]):
            X_processed.append(one_hot_encoded[:, j])
print(f"X_processed: \n{X_processed}")
# Step 2: Combine processed features
X_final = np.column_stack(X_processed)
print(f"X_final: \n{X_final}")

# Step 3: Add intercept term
ones = np.ones((X_final.shape[0], 1))
print(f"ones: \n{ones}")
X_final = np.hstack((ones, X_final))
print(f"X_final: \n{X_final}")

# Step 4: Convert target to numpy array
y = np.array(y, dtype=float).reshape(-1, 1)
print(f"y: \n{y}")
# Step 5: Normal Equation: β = (XᵀX)⁻¹ Xᵀ y
XtX = X_final.T.dot(X_final)
XtX_inv = np.linalg.inv(XtX)
XtY = X_final.T.dot(y)
beta = XtX_inv.dot(XtY)
print(f"beta: \n{beta}")

# Step 6: Predictions
y_pred = X_final.dot(beta)

# Step 7: Evaluation metrics
mse = np.mean((y - y_pred) ** 2)
mae = np.mean(np.abs(y - y_pred))
r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

# Step 8: Return all results
print(f"\nModel Performance:")
print(f"--------------------------------")
print(f"coefficients: {beta.flatten().tolist()}")
print(f"--------------------------------")
print(f"mse: {mse}")
print(f"mae: {mae}")
print(f"r2: {r2}")
print(f"--------------------------------")



