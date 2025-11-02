



# Bias-Variance Tradeoff

7.1 **Definition**

- **Bias:** Error from erroneous or overly simplistic assumptions in the learning algorithm. High bias means the model misses relevant patterns (underfitting).
- **Variance:** Error from sensitivity to small fluctuations in the training set. High variance means the model captures noise as if it were signal (overfitting).

The total expected error for a model can be expressed as:

$$
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

7.2 **Business Impact**

- **High bias:** Model is too simple, underfits data, and performs poorly on both training and test sets. Example: Using a straight line to fit a complex, curved relationship.
- **High variance:** Model is too complex, overfits data, and performs well on training data but poorly on new, unseen data. Example: Using a high-degree polynomial that fits every training point but fails to generalize.
- **Optimal tradeoff:** Achieved when the model is complex enough to capture patterns but simple enough to generalize well to new data.


7.3 **Practical Example: Linear vs. Polynomial Regression**

Let's compare a simple linear model (high bias, low variance) to a high-degree polynomial model (low bias, high variance) on the same data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
x = np.linspace(0, 10, 30)
y = np.sin(x) + np.random.normal(0, 0.3, size=x.shape)
X = x.reshape(-1, 1)

# Linear Regression (high bias)
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Polynomial Regression (degree 12, high variance)
poly = PolynomialFeatures(degree=12)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Plot
plt.scatter(x, y, color='black', label='Data')
plt.plot(x, y_pred_lin, color='blue', label='Linear Fit (High Bias)')
plt.plot(x, y_pred_poly, color='red', label='Poly Fit (High Variance)')
plt.legend()
plt.title('Bias-Variance Tradeoff Example')
plt.show()

# Calculate errors
print(f'Linear Regression MSE: {mean_squared_error(y, y_pred_lin):.3f}')
print(f'Polynomial Regression MSE: {mean_squared_error(y, y_pred_poly):.3f}')
```

- **Interpretation:**
    - The linear model (blue) underfits: it cannot capture the curve, resulting in high bias.
    - The polynomial model (red) overfits: it passes through every point, capturing noise, resulting in high variance.


7.4 **How to Balance Bias and Variance?**

- **Regularization (Lasso, Ridge):** Penalize model complexity to reduce variance without greatly increasing bias.
- **Cross-Validation:** Use to estimate model performance on unseen data and select the best complexity.
- **Ensemble Methods:** Bagging and boosting can reduce variance and/or bias.
- **Increase Training Data:** More data helps complex models generalize better, reducing variance.[^48_4][^48_6]
- **Hyperparameter Tuning:** Adjust model parameters (e.g., tree depth, polynomial degree) to find the optimal tradeoff.


7.5 **Summary Table: Bias-Variance Tradeoff**

| Scenario | Bias | Variance | Typical Error Pattern |
| :-- | :-- | :-- | :-- |
| Underfitting | High | Low | Poor on train \& test data |
| Overfitting | Low | High | Good on train, bad on test |
| Good Tradeoff | Low/Med | Low/Med | Good on both |

**Key Takeaway:**
> The goal is to find a model that is just complex enough to capture the underlying patterns (low bias) but not so complex that it fits the noise (low variance), ensuring strong performance on new, unseen data.[^48_3][^48_5][^48_6][^48_2][^48_4]
