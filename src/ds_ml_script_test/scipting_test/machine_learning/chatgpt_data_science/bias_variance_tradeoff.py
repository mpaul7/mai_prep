import numpy as np
# Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.2, 1.9, 3.0, 3.8, 5.1])

print("X:", X)
print("y:", y)
# Add intercept
X_b = np.hstack([np.ones((X.shape[0],1)), X])
print("X_b:", X_b)
# Linear Regression with L2 (Ridge)

"""Linear Regression with L2 (Ridge)
    X = [[1], [2], [3], [4], [5]]
    y = [1.2, 1.9, 3.0, 3.8, 5.1]
    X_b = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
    lambda_l2 = 0.1
    I = [[1, 0], [0, 1]]
    beta_ridge = [0.1, 0.1]
"""
lambda_l2 = 0.1
I = np.eye(X_b.shape[1])
I[0,0] = 0  # don't regularize bias
beta_ridge = np.linalg.inv(X_b.T.dot(X_b) + lambda_l2*I).dot(X_b.T.dot(y))
print("Ridge coefficients:", beta_ridge)

# Linear Regression with L1 (Lasso) - simplified iterative soft-thresholding
"""Linear Regression with L1 (Lasso) - simplified iterative soft-thresholding
    X = [[1], [2], [3], [4], [5]]
    y = [1.2, 1.9, 3.0, 3.8, 5.1]
    X_b = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
    lambda_l1 = 0.1
    beta_lasso = [0.1, 0.1]
"""
beta_lasso = np.zeros(X_b.shape[1])
lr = 0.01
n_epochs = 1000
lambda_l1 = 0.1
for epoch in range(n_epochs):
    y_pred = X_b.dot(beta_lasso)
    gradient = -2 * X_b.T.dot(y - y_pred) / len(y)
    beta_lasso -= lr * gradient
    # soft-thresholding for L1
    beta_lasso = np.sign(beta_lasso) * np.maximum(0, np.abs(beta_lasso) - lr*lambda_l1)
print("Lasso coefficients:", beta_lasso)
