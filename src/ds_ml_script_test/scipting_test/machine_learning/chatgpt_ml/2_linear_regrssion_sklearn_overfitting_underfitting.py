import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generate non-linear data
np.random.seed(42)
X = np.linspace(0, 5, 100).reshape(-1, 1)
y = 2 * np.sin(X).ravel() + np.random.randn(100) * 0.3  # Nonlinear

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Try different polynomial degrees
for degree in [1, 3, 10]:
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_train_pred = model.predict(X_poly_train)
    y_test_pred = model.predict(X_poly_test)

    print(f"\nDegree {degree}")
    print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    print("Train R²:", r2_score(y_train, y_train_pred))
    print("Test R²:", r2_score(y_test, y_test_pred))

"""_summary_
| Degree | Train RMSE | Test RMSE | Train R² | Test R²-      | Interpretation |
|--------|------------|-----------|----------|---------------|----------------|
| 1      | High       | High      | Low      | Low           | Underfitting   | 
| 3      | Low | Low  | High      | High     | Good Fit      |                |
| 10     | Very Low   | High      | ≈1.0     | Low/Negative  | Overfitting    |
"""

import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', edgecolors='k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
| 10     | Very Low   | High      | ≈1.0     | Low/Negative  | Overfitting    |
"""