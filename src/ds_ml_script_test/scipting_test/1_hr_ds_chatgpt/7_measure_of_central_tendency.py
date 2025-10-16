"""_summary_
Concept:
========

A measure of central tendency is a single value that represents the center or typical value of a dataset.

It helps to summarize a dataset with one representative number.

Common Measures of Central Tendency:
====================================
| Measure           | Formula                           | Description                           |
| ----------------- | --------------------------------- | ------------------------------------- |
| Mean              | Σx/n                              | Average value                         |
| Weighted Mean     | Σ(x_i * w_i) / Σw_i              | Average value with weights            |
| Median            | Middle value of sorted data       | Middle value of sorted data          |
| Mode              | Most frequent value               | Value that appears most often         |
| Geometric Mean    | (x_1 * x_2 * ... * x_n)^(1/n)    | Average rate of growth               |
| Harmonic Mean     | n / Σ(1/x_i)                     | Inverse of average of inverses        |
| Trimmed Mean      | Average of middle 80% of data    | Average of middle 80% of data        |
| Interquartile Mean| Average of Q1 and Q3             | Average of 25th and 75th percentiles  |

Notes & HackerRank Relevance:
============================
1. Mean → Average value
2. Weighted Mean → Average value with weights
3. Median → Middle value of sorted data
4. Mode → Value that appears most often
5. Geometric Mean → Average rate of growth
6. Harmonic Mean → Inverse of average of inverses
7. Trimmed Mean → Average of middle 80% of data
8. Interquartile Mean → Average of 25th and 75th percentiles

HackerRank tasks often require:
==============================
1. Compute mean, median, mode, geometric mean, harmonic mean, trimmed mean, interquartile mean
2. Compute weighted mean
3. Compute median, mode, trimmed mean, interquartile mean
4. Compute geometric mean, harmonic mean
5. Compute trimmed mean, interquartile mean
6. Compute weighted mean
7. Compute median, mode, trimmed mean, interquartile mean
8. Compute geometric mean, harmonic mean


Importance:
==========

1. Mean → sensitive to outliers

2. Median → robust to outliers

3. Mode → useful for categorical data

4. Geometric Mean → useful for growth rates

5. Harmonic Mean → useful for rates

6. Trimmed Mean → useful for removing outliers

7. Interquartile Mean → useful for removing outliers

Example:
==========
Data: 1, 2, 2, 3, 100

Mean = 21.6 → affected by outlier 100

Median = 2 → better represents center

Mode = 2 → most frequent
"""
import numpy as np
from collections import Counter

def calculate_mean(data):
    return round(np.mean(data), 2)

def calculate_median(data):
    return round(np.median(data), 2)

def calculate_mode(data):
    counts = Counter(data)
    max_count = max(counts.values())
    mode_values = [k for k, v in counts.items() if v == max_count]
    # If multiple modes, return smallest value
    return min(mode_values)

if __name__ == "__main__":
    # Input: space-separated numbers
    input_str = "1 2 2 3 100"
    data = list(map(float, input_str.split()))
    
    mean = calculate_mean(data)
    median = calculate_median(data)
    mode = calculate_mode(data)
    
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")

"""_summary_
Key Notes
=========
1. Mean → best for symmetric distributions
2. Median → best for skewed distributions or with outliers
3. Mode → best for categorical or discrete data
4. Geometric Mean → best for growth rates
5. Harmonic Mean → best for rates
6. Trimmed Mean → best for removing outliers
7. Interquartile Mean → best for removing outliers

Multiple modes → often return smallest or all
"""