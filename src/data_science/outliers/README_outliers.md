# Outlier Detection and Analysis

This module provides comprehensive tools for outlier detection and analysis, specifically designed for data science interview preparation.

## Overview

Outliers are observations that lie an abnormal distance from other values in a dataset. Understanding how to detect, analyze, and handle outliers is crucial for data scientists and is frequently tested in technical interviews.

## Key Concepts Covered

### 1. Statistical Methods
- **Z-Score Method**: Uses standard deviations from the mean
- **Modified Z-Score**: More robust using median and MAD
- **IQR Method**: Uses interquartile range (most common)
- **Percentile Method**: Uses percentile thresholds

### 2. Machine Learning Methods
- **Isolation Forest**: Tree-based anomaly detection
- **Local Outlier Factor (LOF)**: Density-based detection

### 3. Impact Analysis
- Effect on mean, median, standard deviation
- Robust vs non-robust statistics
- Data quality assessment

## Files Structure

```
src/outliers/
├── outliers.py              # Main outlier detection toolkit
├── test_outliers.py         # Comprehensive test suite
├── outlier_visualization.py # Visualization tools
└── README_outliers.md       # This documentation
```

## Quick Start

### Basic Usage

```python
from outliers import OutlierDetector

# Initialize detector
detector = OutlierDetector()

# Sample data with outliers
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]

# Detect outliers using IQR method
iqr_result = detector.iqr_outliers(data)
print(f"Outliers: {iqr_result['outliers']}")  # [100]

# Compare multiple methods
comparison = detector.compare_methods(data)
print(comparison)
```

### HackerRank-Style Problems

```python
from outliers import hackerrank_outlier_problems

# Run practice problems
hackerrank_outlier_problems()
```

### Visualization

```python
from outlier_visualization import OutlierVisualizer

visualizer = OutlierVisualizer()
visualizer.plot_outlier_detection_comparison(data)
```

## Common Interview Questions

### 1. "How do you detect outliers?"

**Answer**: There are several methods:

1. **IQR Method** (most common):
   - Calculate Q1 (25th percentile) and Q3 (75th percentile)
   - IQR = Q3 - Q1
   - Outliers: values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR

2. **Z-Score Method**:
   - Calculate Z-score = (x - mean) / std_dev
   - Outliers: |Z-score| > 2 or 3 (depending on threshold)

3. **Modified Z-Score** (more robust):
   - Uses median and MAD instead of mean and std_dev

### 2. "What's the difference between outliers and anomalies?"

**Answer**: 
- **Outliers**: Statistical concept - points that deviate significantly from the distribution
- **Anomalies**: Domain-specific concept - unusual patterns that might indicate problems or interesting events
- All anomalies are outliers, but not all outliers are anomalies

### 3. "How do outliers affect different statistics?"

**Answer**:
- **Mean**: Very sensitive to outliers
- **Median**: Robust to outliers
- **Standard Deviation**: Very sensitive to outliers
- **MAD (Median Absolute Deviation)**: Robust to outliers

### 4. "When should you remove outliers?"

**Answer**:
- **Remove when**: Data entry errors, measurement errors, not from target population
- **Keep when**: Legitimate extreme values, rare but valid events, small datasets
- **Always**: Investigate before deciding

## Method Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| IQR | Simple, robust, interpretable | Fixed threshold, assumes distribution | General purpose |
| Z-Score | Easy to understand | Sensitive to outliers in calculation | Normal distributions |
| Modified Z-Score | More robust than Z-score | Still assumes distribution | Skewed distributions |
| Isolation Forest | No distribution assumptions | Black box, needs tuning | Complex datasets |
| LOF | Good for local outliers | Computationally expensive | Clustered data |

## Practice Problems

The module includes 10+ HackerRank-style problems covering:

1. Basic outlier counting
2. IQR implementation
3. Impact analysis
4. Robust statistics comparison
5. Outlier percentage calculation
6. Bootstrap outlier detection
7. Moving window outliers
8. Multivariate outliers
9. Outlier visualization
10. Method comparison

## Running Tests

```bash
# Run all tests
python test_outliers.py

# Run specific test class
python -m unittest test_outliers.TestOutlierDetector

# Run with verbose output
python test_outliers.py -v
```

## Visualization Examples

The visualization module provides:

1. **Method Comparison Plot**: Compare all methods side-by-side
2. **Boxplot with Boundaries**: Show IQR boundaries and outliers
3. **Distribution Analysis**: Histogram and Q-Q plots
4. **Impact Analysis**: Before/after outlier removal

## Tips for Interviews

### Code Implementation Tips
1. Always handle edge cases (empty data, single point, etc.)
2. Use appropriate statistical measures (sample vs population)
3. Consider computational efficiency for large datasets
4. Provide clear documentation and examples

### Conceptual Tips
1. Understand the business context before removing outliers
2. Know when to use robust vs non-robust methods
3. Be able to explain trade-offs between methods
4. Understand the impact on downstream analysis

### Common Mistakes to Avoid
1. Removing outliers without investigation
2. Using mean-based methods on skewed data
3. Not considering multivariate outliers
4. Forgetting to document outlier handling decisions

## Advanced Topics

For senior positions, be prepared to discuss:

1. **Multivariate Outliers**: Mahalanobis distance, PCA-based detection
2. **Time Series Outliers**: Seasonal decomposition, change point detection
3. **Streaming Data**: Online outlier detection algorithms
4. **Ensemble Methods**: Combining multiple detection methods
5. **Domain-Specific**: Fraud detection, network intrusion, quality control

## Resources for Further Learning

1. **Books**: 
   - "Outlier Analysis" by Charu Aggarwal
   - "Robust Statistics" by Huber & Ronchetti

2. **Papers**:
   - Isolation Forest (Liu et al., 2008)
   - Local Outlier Factor (Breunig et al., 2000)

3. **Online Courses**:
   - Coursera: Anomaly Detection courses
   - edX: Statistical Learning courses

## Contributing

To add new outlier detection methods or problems:

1. Add the method to `OutlierDetector` class
2. Create corresponding tests in `test_outliers.py`
3. Add visualization if applicable
4. Update this documentation

## License

This educational material is provided under the MIT License for interview preparation purposes.

