"""_summary_
Concept
========
1. Outliers are values that deviate significantly from the majority of data.
2. Can skew mean, variance, and correlation

3. Important to detect for data cleaning and analysis

Outlier detection is the process of identifying data points that deviate significantly from the rest of the data.

Common Methods of Outlier Detection:
====================================
| Method           | Formula                           | Description                           |
| ----------------- | --------------------------------- | ------------------------------------- |
| IQR Method       | Q3 - Q1                           | Difference between 75th and 25th percentiles |
| Z-Score Method   | (x - mean) / standard deviation   | Standardized deviation from the mean  |

Common Methods

IQR Method (robust, non-parametric)

1. Outliers are outside:
 q1 - 1.5 * iqr and q3 + 1.5 * iqr
2. z-score (parametric, assumes normality   )
 |z| > 3
 z = (x - mean) / standard deviation
"""

import numpy as np

def detect_outliers_iqr(data):
    """
    Detect outliers using IQR method
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower or x > upper]
    return outliers

def detect_outliers_zscore(data, threshold=1.5):
    """
    Detect outliers using Z-score method
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    outliers = [x for x in data if abs((x - mean)/std) > threshold]
    return outliers

if __name__ == "__main__":
    input_str = "10 12 12 13 15 12 14 100 110"
    data = list(map(float, input_str.split()))
    arr = np.array(data)
    
    outliers_iqr = detect_outliers_iqr(arr)
    outliers_z = detect_outliers_zscore(arr)
    
    print(f"Outliers (IQR method): {outliers_iqr}")
    print(f"Outliers (Z-score method): {outliers_z}")
