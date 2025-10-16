import numpy as np
# Example true and predicted labels
y_true = np.array([0, 1, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1])

# Confusion matrix elements
TP = np.sum((y_true == 1) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))

# Accuracy
accuracy = (TP + TN) / len(y_true)

# Precision
precision = TP / (TP + FP) if (TP + FP) != 0 else 0

# Recall
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

# F1-score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print("\nClassification Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
