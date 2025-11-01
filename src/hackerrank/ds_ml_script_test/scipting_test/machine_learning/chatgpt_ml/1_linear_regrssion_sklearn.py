# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ====================================
# Step 1: Generate Synthetic Regression Data
# ====================================
# 100 samples, 3 features, with noise
X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
print(f"X: {X}")
print(f"y: {y}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====================================
# Step 2: Train Linear Regression Model
# ====================================
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# ====================================
# Step 3: Compute Evaluation Metrics
# ====================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ====================================
# Step 4: Display Results
# ====================================
print("\nModel Coefficients and Intercept:")
print("--------------------------------")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

print("\nModel Evaluation Metrics:")
print("--------------------------------")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

print("\nModel Interpretation:")
print("--------------------------------")
print("Radio has a stronger influence than TV and Newspaper.")
print("Negative coefficient for Newspaper may mean it adds little or even noise.")  

print("\nModel Evaluation Metrics:")
print("--------------------------------")
print(f"Mean Absolute Error (MAE): {mae:.3f}, On average, predictions are off by ~8 units.")
print(f"Mean Squared Error (MSE): {mse:.3f}, Average squared deviation = 92.6 (higher due to squaring).")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}, On average, model predictions deviate by ~9.6 units from actual.")
print(f"R² Score: {r2:.3f}, Model explains 98.3% of variance → excellent fit.")



"""_summary_

model.coef_ → shows how much each feature contributes to Sales.

model.intercept_ → base sales when all ad spend = 0.

R² score → model accuracy (1.0 = perfect fit).

MAE = 7.99	On average, predictions are off by ~8 units.	
MSE = 92.63	Average squared deviation = 92.6 (higher due to squaring).	
RMSE = 9.63	On average, model predictions deviate by ~9.6 units from actual.	
R² = 0.983	Model explains 98.3% of variance → excellent fit.
"""

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='blue', edgecolors='k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()


"""
1. UNDERFITTING (Model too simple)
 Example:

You fit a linear model on nonlinear data (e.g., house prices vs square footage that actually has a curve).

 Behavior of Metrics
Metric	What Happens	Reason
MAE	High	Model is far from actual values for most points
MSE	High	Same reason; squaring increases penalty
RMSE	High	Same as MSE, large overall deviation
R²	Low (close to 0 or negative)	Model explains very little of the data variance
Interpretation:

Predictions are consistently off.

The line cannot capture underlying relationships.

Model lacks complexity (too simple, too few parameters or features).

Fix:

Add more features (polynomial, interactions).

Use a more flexible model (decision tree, random forest, etc.).    
"""

"""
2. OVERFITTING (Model too complex)
 Example:

You fit a 15th-degree polynomial on simple data with only 10 points.
Model fits training data perfectly — even noise.

Behavior of Metrics
Metric	Training Data	Test Data	Reason
MAE / MSE / RMSE	Very Low	High	Model memorizes training data but fails to generalize
R²	≈ 1.0	Low or Negative	Perfect on training, poor on unseen data
Interpretation:

Model captures noise rather than patterns.

Very small training errors but large unseen (test) errors.

Fix:

Use regularization (L1, L2 → Ridge, Lasso).

Use cross-validation.

Simplify model (reduce features or model depth).
"""

"""

3. IDEAL MODEL (Balanced Fit)
Metric	Value	Meaning
MAE, MSE, RMSE	Low and similar for train/test	Good fit, consistent performance
R²	High on both train/test (≈ 0.8–0.95)	Captures variance without memorizing noise

Model generalizes well → this is what you want on HackerRank or real-world tasks.
"""