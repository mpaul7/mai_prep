# Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ====================================
# Step 1: Load dataset
# ====================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# ====================================
# Step 2: Split data
# ====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================
# Step 3: Standardize features
# ====================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================================
# Step 4: Train logistic regression model
# ====================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ====================================
# Step 5: Predictions
# ====================================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1

# ====================================
# Step 6: Evaluation Metrics
# ====================================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

# ====================================
# Step 7: Display results
# ====================================
print("Multiple Logistic Regression Results:")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"ROC-AUC:   {roc:.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


"""
Interpretation of Metrics
========================================================================================================================
Metric	    | Meaning	                                                 | Physical Significance
-------------------------------------------------------------------------------------------------------------------------
Accuracy	| % of total correct predictions	                         |  High accuracy → model performs well overall
-----------------------------------------------------------------------------------------------------------------------
Precision	| Correctly predicted positives / all predicted positives	 | High precision → few false positives
-----------------------------------------------------------------------------------------------------------------------
Recall	    | Correctly predicted positives / all actual positives	     | High recall → few false negatives
----------------------------------------------------------------------------------------------------
F1-Score	| Harmonic mean of precision and recall	                     | Good balance between precision & recall
-----------------------------------------------------------------------------------------------------------------------
ROC-AUC	    | Area under ROC curve (probability threshold curve)	     | Measures separability — higher = better discrimination between classes
-----------------------------------------------------------------------------------------------------------------------
"""

"""
Common Use Cases
========================================================================================================================
1. Image classification (cat / dog / bird)
2. Sentiment analysis (positive / neutral / negative)
3. Product recommendation (buy / not buy)
4. Medical diagnosis (disease / healthy)
"""

"""
Logistic Regression Coefficients Recap:

For multiple logistic regression:

logit(p) = ln(p/(1-p)) = β0 + β1*X1 + β2*X2 + ... + βn*Xn

Where:
- p = probability of outcome = 1
- βi = coefficient for feature Xi

Interpretation of β values:
- Positive β → increases log-odds → higher probability of outcome = 1
- Negative β → decreases log-odds → lower probability of outcome = 1
- Magnitude of β → strength of effect

Odds ratio:
- OR = e^βi
- Each one-unit increase in Xi multiplies the odds by OR

Example:

Suppose a feature radius_mean has coefficient 

Odds Ratio = e^0.7 ≈ 2.01

Interpretation:
For every one-unit increase in radius_mean, the odds of having cancer (class 1) double (≈2x), 
holding other features constant.
"""