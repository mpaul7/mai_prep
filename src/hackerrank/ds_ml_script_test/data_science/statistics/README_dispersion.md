# Dispersion Metrics Analysis

This module provides comprehensive tools for calculating and analyzing dispersion metrics, specifically designed for data science interview preparation.

## Overview

Dispersion metrics measure how spread out or scattered data points are around a central value. Understanding these metrics is crucial for data scientists and is frequently tested in technical interviews.

## Key Concepts Covered

### 1. Basic Dispersion Measures
- **Range**: Simple difference between max and min values
- **Interquartile Range (IQR)**: Spread of the middle 50% of data
- **Variance**: Average squared deviation from the mean
- **Standard Deviation**: Square root of variance (same units as data)

### 2. Advanced Measures
- **Coefficient of Variation (CV)**: Relative variability measure
- **Mean Absolute Deviation (MAD)**: Average absolute deviation
- **Quartile Coefficient of Dispersion**: Relative measure using quartiles

### 3. Robust vs Non-Robust Measures
- **Robust**: IQR, MAD from median (resistant to outliers)
- **Non-Robust**: Range, Standard Deviation, Variance (sensitive to outliers)

## Files Structure

```
src/statistics/
├── dispersion.py              # Main dispersion calculation toolkit
├── test_dispersion.py         # Comprehensive test suite
├── dispersion_visualization.py # Visualization tools
└── README_dispersion.md       # This documentation
```

## Quick Start

### Basic Usage

```python
from dispersion import DispersionCalculator

# Initialize calculator
calc = DispersionCalculator()

# Sample data
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Calculate IQR
iqr_result = calc.calculate_iqr(data)
print(f"IQR: {iqr_result['iqr']}")  # 4.5

# Calculate standard deviation
std_result = calc.calculate_standard_deviation(data, sample=True)
print(f"Sample Std Dev: {std_result['std_dev']:.2f}")  # 3.03

# Compare all measures
comparison = calc.compare_dispersion_measures(data)
print(comparison)
```

### HackerRank-Style Problems

```python
from dispersion import hackerrank_dispersion_problems

# Run practice problems
hackerrank_dispersion_problems()
```

### Visualization

```python
from dispersion_visualization import DispersionVisualizer

visualizer = DispersionVisualizer()
datasets = [("Dataset 1", data1), ("Dataset 2", data2)]
visualizer.plot_dispersion_comparison(datasets)
```

## Common Interview Questions

### 1. "What's the difference between variance and standard deviation?"

**Answer**: 
- **Variance**: Average of squared deviations from the mean (units²)
- **Standard Deviation**: Square root of variance (original units)
- **Why both exist**: Variance is mathematically convenient; std dev is more interpretable
- **Formula**: σ = √(σ²)

### 2. "When would you use sample vs population variance?"

**Answer**:
- **Population variance**: When you have the entire population (denominator: n)
- **Sample variance**: When you have a sample (denominator: n-1)
- **Why n-1**: Bessel's correction - provides unbiased estimate of population variance
- **Sample variance is always larger** than population variance for the same data

### 3. "What is the Coefficient of Variation and when is it useful?"

**Answer**:
- **Formula**: CV = (Standard Deviation / Mean) × 100%
- **Purpose**: Compare variability between datasets with different scales/units
- **Example**: Comparing stock price volatility ($100 stock vs $10 stock)
- **Interpretation**: <15% (low), 15-35% (moderate), >35% (high variability)

### 4. "How do outliers affect different dispersion measures?"

**Answer**:
- **Highly sensitive**: Range, Standard Deviation, Variance
- **Robust (resistant)**: IQR, MAD from median
- **Why**: Robust measures use percentiles/medians instead of means
- **Practical impact**: Choose robust measures when outliers are present

### 5. "What is IQR and how is it calculated?"

**Answer**:
- **Definition**: Interquartile Range = Q3 - Q1
- **Q1**: 25th percentile, **Q3**: 75th percentile
- **Interpretation**: Spread of the middle 50% of data
- **Outlier detection**: Values outside [Q1-1.5×IQR, Q3+1.5×IQR] are potential outliers

## Formulas Reference

### Sample vs Population Formulas

| Measure | Sample Formula | Population Formula |
|---------|---------------|-------------------|
| Variance | Σ(x-x̄)²/(n-1) | Σ(x-μ)²/n |
| Std Dev | √[Σ(x-x̄)²/(n-1)] | √[Σ(x-μ)²/n] |

### Other Important Formulas

- **Range**: max(x) - min(x)
- **IQR**: Q3 - Q1
- **CV**: (σ/μ) × 100%
- **MAD**: Σ|x-center|/n

## Practice Problems

The module includes 5+ HackerRank-style problems covering:

1. **Basic Range Calculation**: Simple max-min difference
2. **IQR Implementation**: Quartile calculation and IQR
3. **Sample vs Population Variance**: Understanding the difference
4. **Coefficient of Variation**: Comparing relative variability
5. **Robust vs Non-Robust**: Impact of outliers on measures

## Method Comparison

| Measure | Formula | Robust? | Units | Best For |
|---------|---------|---------|-------|----------|
| Range | max - min | No | Original | Quick overview |
| IQR | Q3 - Q1 | Yes | Original | Outlier-resistant spread |
| Variance | Σ(x-μ)²/n | No | Squared | Mathematical calculations |
| Std Dev | √Variance | No | Original | Interpretable spread |
| CV | σ/μ × 100% | No | Percentage | Comparing different scales |
| MAD | Σ\|x-median\|/n | Yes | Original | Robust alternative to std dev |

## Running Tests

```bash
# Run all tests
python test_dispersion.py

# Run specific test class
python -m unittest test_dispersion.TestDispersionCalculator

# Run with verbose output
python test_dispersion.py -v
```

## Visualization Examples

The visualization module provides:

1. **Dispersion Comparison**: Compare all measures across datasets
2. **Variance vs Std Dev**: Understand the relationship
3. **Robust vs Non-Robust**: Impact of outliers
4. **Coefficient of Variation**: Scale-independent comparison

## Tips for Interviews

### Conceptual Understanding
1. **Know when to use each measure** (robust vs non-robust)
2. **Understand the units** (original vs squared)
3. **Explain the intuition** behind each formula
4. **Discuss practical applications** in data science

### Implementation Tips
1. **Handle edge cases** (empty data, single point, identical values)
2. **Use appropriate formulas** (sample vs population)
3. **Consider computational efficiency** for large datasets
4. **Validate results** with known examples

### Common Mistakes to Avoid
1. Using population formula when you have a sample
2. Forgetting that variance is in squared units
3. Not considering outliers when choosing measures
4. Misinterpreting coefficient of variation

## Advanced Topics

For senior positions, be prepared to discuss:

1. **Multivariate Dispersion**: Covariance matrices, Mahalanobis distance
2. **Time Series Dispersion**: Rolling variance, volatility clustering
3. **Robust Estimators**: Trimmed variance, Winsorized variance
4. **Bootstrap Confidence Intervals**: For dispersion estimates
5. **Bayesian Approaches**: Prior distributions for variance parameters

## Real-World Applications

### Finance
- **Volatility measurement**: Standard deviation of returns
- **Risk assessment**: Coefficient of variation for comparing investments
- **Value at Risk**: Using quantiles and dispersion measures

### Quality Control
- **Process variation**: Control charts using standard deviation
- **Six Sigma**: Reducing process variation
- **Capability indices**: Relating specification limits to process variation

### Machine Learning
- **Feature scaling**: Understanding data spread before normalization
- **Outlier detection**: Using IQR and robust measures
- **Model validation**: Variance in cross-validation scores

## Resources for Further Learning

1. **Books**:
   - "Introduction to Mathematical Statistics" by Hogg & Craig
   - "Robust Statistics" by Huber & Ronchetti

2. **Online Resources**:
   - Khan Academy: Statistics and Probability
   - Coursera: Statistical Inference courses

3. **Practice Platforms**:
   - HackerRank: Statistics domain
   - LeetCode: Statistics problems
   - Kaggle Learn: Statistics courses

## Contributing

To add new dispersion measures or problems:

1. Add the method to `DispersionCalculator` class
2. Create corresponding tests in `test_dispersion.py`
3. Add visualization if applicable
4. Update this documentation

## License

This educational material is provided under the MIT License for interview preparation purposes.
