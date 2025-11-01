"""_summary_
Concept
========
Measures of dispersion tell us how spread out the data is.

Common Measures of Dispersion:
====================================
| Measure                            | Formula                    | Description                                  |
|----------------------------------  |----------------------------|--------------------------------------------  | 
| Range                              | Max - Min                  | Difference between max and min               |
| Interquartile Range                | Q3 - Q1                    | Difference between 75th and 25th percentiles |
| Variance                           | Σ(x-x̄)²/(n-1)              | Average squared deviation from the mean      |
| Standard Deviation                 | √[Σ(x-x̄)²/(n-1)]           | Square root of variance                      |
| Coefficient of Variation           | (σ/μ) × 100%               | Relative variability measure                 |
| Mean Absolute Deviation            | Σ|x-center|/n              | Average absolute deviation from the center   |
| Quartile Coefficient of Dispersion | (Q3-Q1)/(Q3+Q1)            | Relative measure using quartiles             |


Notes & HackerRank Relevance:
============================
1. Range → Simple difference between max and min values
2. Interquartile Range → Spread of the middle 50% of data


| Metric | Sensitivity                | Use Case                         |
| ------ | -------------------------- | -------------------------------- |
| Range  | Very sensitive to outliers | Quick overview of spread         |
| IQR    | Robust to outliers         | Detecting variability & outliers |

"""

import numpy as np


input_str = "10 2 5 8 7 3 15 20"
data = list(map(float, input_str.split()))
arr = np.array(data)

print(f"\n\nCalculating Range:")
range_val = round(np.max(arr) - np.min(arr), 2)
print(f"Range: {range_val}")

print(f"\n\nCalculating IQR:")
q1 = np.percentile(arr, 25)
print(f"Q1: {q1}")
q3 = np.percentile(arr, 75)
print(f"Q3: {q3}")
iqr_val = round(q3 - q1, 2)
print(f"IQR: {iqr_val}")

