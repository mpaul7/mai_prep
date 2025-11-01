import numpy as np
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Covariance
cov_matrix = np.cov(X, Y, bias=False)
print("Covariance matrix:\n", cov_matrix)

# Correlation
corr_matrix = np.corrcoef(X, Y)
print("Correlation coefficient:\n", corr_matrix)
