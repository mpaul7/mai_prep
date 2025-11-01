import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np

# Generate sample data
X = np.linspace(0, 5, 20).reshape(-1,1)
y = 2 * X**2 + np.random.randn(20,1) * 2

degrees = [1, 2, 10]
train_errors, test_errors = [], []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    X_train, X_test = X_poly[:15], X_poly[15:]
    y_train, y_test = y[:15], y[15:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

plt.plot(degrees, train_errors, label='Train Error', marker='o')
plt.plot(degrees, test_errors, label='Test Error', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('Overfitting vs Underfitting')
plt.legend()
plt.show()
