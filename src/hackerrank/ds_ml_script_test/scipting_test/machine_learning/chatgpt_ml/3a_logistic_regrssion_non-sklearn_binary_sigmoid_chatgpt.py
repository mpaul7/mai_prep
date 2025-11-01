import numpy as np

def logistic_regression_binary(X, y, lr=0.01, epochs=5):
    """
    Logistic Regression (Binary Classification) from scratch.
    Two features only.
    
    Parameters:
        X : list of [x1, x2]
        y : list of binary targets (0 or 1)
        lr : learning rate
        epochs : number of iterations for gradient descent
    Returns:
        weights, bias, predictions, and metrics
    """
    # Convert input to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1, 1)
    
    print(f"X: {X}")
    print(f"y: {y}")
    
    n_samples, n_features = X.shape
    print(f"n_samples: {n_samples}")
    print(f"n_features: {n_features}")
    
    # Initialize weights and bias
    weights = np.zeros((n_features, 1))
    bias = 0.0

    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Training loop - Gradient Descent
    for epoch in range(epochs):
        # Linear model
        """_summary_
        z = X * weights + bias
        z = w1*x1 + w2*x2 + b
        X → matrix of input features
        w → vector of weights (model parameters)
        b → bias term (scalar or vector added to all samples)
        z → linear combination (sometimes called logit in logistic regression)
        Args:
            z (np.ndarray): Linear model
        """
        linear_model = np.dot(X, weights) + bias
        # print(f"linear_model: {linear_model}")
        # Prediction
        y_pred = sigmoid(linear_model)

        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Update parameters
        weights -= lr * dw
        bias -= lr * db

    # Final predictions
    linear_model = np.dot(X, weights) + bias
    print(f"linear_model: {linear_model}")
    y_pred_prob = sigmoid(linear_model)
    print(f"y_pred_prob: {y_pred_prob}")
    y_pred_label = (y_pred_prob >= 0.5).astype(int)
    print(f"y_pred_label: {y_pred_label}")

    # Metrics
    accuracy = np.mean(y_pred_label == y) * 100
    mse = np.mean((y - y_pred_prob) ** 2)
    mae = np.mean(np.abs(y - y_pred_prob))

    print(f"weights: {weights.flatten().tolist()}")
    print(f"--------------------------------")
    print(f"bias: {float(bias)}")
    print(f"mse: {mse}")
    print(f"mae: {mae}")
    print(f"accuracy: {accuracy}")
    print(f"--------------------------------")
    print(f"predicted_labels: {y_pred_label.flatten().tolist()}")
    print(f"predicted_probabilities: {y_pred_prob.flatten().tolist()}")
    print(f"--------------------------------")
    print(f"predicted_probabilities: {y_pred_prob.flatten().tolist()}")
    print(f"--------------------------------")



# Example data (2 features, binary output)
X = [
    [2.5, 1.7],
    [1.3, 3.1],
    [3.3, 4.0],
    [4.0, 2.8],
    [3.8, 5.0],
    [2.2, 1.0],
    [5.1, 3.2],
    [4.5, 2.3],
]

y = [0, 0, 1, 1, 1, 0, 1, 1]

# Train logistic regression
result = logistic_regression_binary(X, y, lr=0.1, epochs=5)
print(result)
