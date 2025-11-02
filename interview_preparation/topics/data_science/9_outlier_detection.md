

---

# Outlier Detection \& Robust Statistics

## 9.1 **Outlier Detection**

### **Definition**

Outlier detection is the process of identifying data points that deviate significantly from the majority of the data. Outliers can arise from data entry errors, measurement variability, or rare but important events.

### **Common Methods**

- **Z-Score Method:** Measures how many standard deviations a point is from the mean. Points with |z| > 3 are often considered outliers.

```python
import numpy as np
data = np.array([1, 2, 2, 3, 1, 3, 10])
z_scores = (data - np.mean(data)) / np.std(data)
outliers = data[np.abs(z_scores) > 2]
print("Outliers (z-score):", outliers)
```

- **Interquartile Range (IQR) Method:** Outliers are points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

```python
import numpy as np
data = np.array([1, 2, 2, 3, 1, 3, 10])
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = data[(data < lower) | (data > upper)]
print("Outliers (IQR):", outliers)
```

- **Visual Methods:** Box plots, scatter plots, and histograms can quickly reveal outliers.
- **Distance-Based \& Machine Learning Methods:** DBSCAN, Isolation Forest, One-Class SVM, and PCA-based methods are used for high-dimensional or complex data.


### **Business Impact**

- Outliers can distort model estimates, leading to poor predictions and business decisions.
- In finance, outliers may indicate fraud or market shocks.
- In manufacturing, outliers can signal defects or process failures.
- In healthcare, outliers may reveal rare but critical patient conditions.

***

## 9.2 **Robust Statistics**

### **Definition**

Robust statistics are methods that are less sensitive to outliers and non-normal data. They provide more reliable estimates when data contains anomalies.

### **Common Robust Methods**

- **Median:** Unlike the mean, the median is not affected by extreme values.
- **Median Absolute Deviation (MAD):** A robust measure of spread.
- **Robust Regression:** Techniques like RANSAC or Huber regression reduce the influence of outliers on model fitting.

```python
from sklearn.linear_model import HuberRegressor
model = HuberRegressor().fit(X, y)
```

- **Trimmed Means:** Calculate the mean after removing a percentage of the highest and lowest values.


### **Business Impact**

- Using robust methods prevents a few extreme values from skewing results, leading to more trustworthy insights and decisions.
- In pricing, robust statistics help avoid overreacting to rare, extreme sales.
- In quality control, robust methods ensure that a few defective items do not distort process metrics.

***

## 9.3 **Summary Table: Outlier Detection \& Robust Statistics**

| Method/Concept | Purpose | Business Impact |
| :-- | :-- | :-- |
| Z-Score/IQR | Identify outliers | Prevent model distortion, spot fraud |
| Visual/ML Methods | Detect complex outliers | Find rare events, improve reliability |
| Median/MAD | Robust central tendency/spread | Reliable stats with anomalies |
| Robust Regression | Outlier-resistant modeling | Stable predictions, better decisions |


***

**Key Takeaway:**
Outlier detection and robust statistics are essential for high-quality data analysis and modeling. They protect your business from making decisions based on rare errors or anomalies, ensuring more accurate and actionable insights.
