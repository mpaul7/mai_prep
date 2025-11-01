# Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ======================
# Step 1: Load dataset
# ======================
data = load_breast_cancer()
X = data.data
y = data.target

# ======================
# Step 2: Split dataset
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Step 3: Train model
# ======================
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# ======================
# Step 4: Predictions
# ======================
y_pred = model.predict(X_test)

# ======================
# Step 5: Evaluate model
# ======================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



print("\nInterpretation:")
print("--------------------------------")

print("Accuracy = 96% → Model correctly predicts 96% of samples.")

print("Precision (for class 1) = 0.99 → 99% of predicted positives are true positives.")

print("Recall (for class 1) = 0.96 → Model captures 96% of actual positives.")

print("F1-score → Harmonic mean of precision & recall (balances the two).")


"""_summary_
Evaluation Metrics for Logistic Regression
==========================================
Metric	             | Description	                            | Best When
-----------------------------------------------------------------------------------------------------
Accuracy	         | % of correct predictions	                | Balanced classes
-----------------------------------------------------------------------------------------------------
Precision	         | How many predicted positives are true	| When false positives are costly
Recall (Sensitivity) | How many actual positives are caught	    | When false negatives are costly
F1-Score	         | Balance of Precision & Recall	        | Imbalanced data
-----------------------------------------------------------------------------------------------------
ROC-AUC	             | Area under the ROC curve	                | Overall performance (probabilistic)
Log Loss	         | Penalizes wrong probabilities	        | For probabilistic evaluation
"""

"""
Common Use Cases
==========================================
1. Spam detection (spam / not spam)
2. Customer churn (leave / stay)
3. Credit risk (default / no default)
4. Medical diagnosis (disease / healthy)

"""