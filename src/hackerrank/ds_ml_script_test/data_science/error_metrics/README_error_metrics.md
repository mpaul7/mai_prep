# Error Metrics Analysis

This module provides comprehensive tools for calculating and analyzing error metrics, specifically designed for data science interview preparation.

## Overview

Error metrics are essential for evaluating model performance in machine learning and statistics. Understanding when to use different metrics and how to interpret them is crucial for data scientists and is frequently tested in technical interviews.

## Key Concepts Covered

### 1. Basic Error Metrics
- **Mean Squared Error (MSE)**: Average of squared differences
- **Root Mean Squared Error (RMSE)**: Square root of MSE (same units as target)
- **Mean Absolute Error (MAE)**: Average of absolute differences
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error metric

### 2. Advanced Error Metrics
- **Symmetric Mean Absolute Percentage Error (SMAPE)**: Symmetric version of MAPE
- **Mean Absolute Scaled Error (MASE)**: Scale-independent error metric
- **Mean Squared Logarithmic Error (MSLE)**: For positive values, penalizes underestimation
- **Mean Directional Accuracy (MDA)**: Measures directional prediction accuracy

### 3. Goodness of Fit Metrics
- **R-squared (R²)**: Proportion of variance explained
- **Adjusted R-squared**: R² adjusted for number of features
- **Theil's U Statistic**: Inequality coefficient for forecast evaluation

## Files Structure

```
src/error_metrics/
├── error_metrics.py          # Main error metrics calculation toolkit
├── test_error_metrics.py     # Comprehensive test suite
├── error_visualization.py    # Visualization tools
└── README_error_metrics.md   # This documentation
```

## Quick Start

### Basic Usage

```python
from error_metrics import ErrorMetricsCalculator

# Initialize calculator
calc = ErrorMetricsCalculator()

# Sample data
y_true = [1, 2, 3, 4, 5]
y_pred = [1.1, 2.1, 2.9, 4.1, 4.9]

# Calculate MSE
mse = calc.mean_squared_error(y_true, y_pred)
print(f"MSE: {mse['mse']:.4f}")  # 0.0200

# Calculate RMSE
rmse = calc.root_mean_squared_error(y_true, y_pred)
print(f"RMSE: {rmse['rmse']:.4f}")  # 0.1414

# Calculate all metrics
all_metrics = calc.calculate_all_metrics(y_true, y_pred)
print(all_metrics)
```

### HackerRank-Style Problems

```python
from error_metrics import hackerrank_error_metrics_problems

# Run practice problems
hackerrank_error_metrics_problems()
```

### Visualization

```python
from error_visualization import ErrorMetricsVisualizer

visualizer = ErrorMetricsVisualizer()
visualizer.plot_error_analysis(y_true, y_pred, "Error Analysis")
```

## Common Interview Questions

### 1. "What's the difference between MSE and MAE?"

**Answer**:
- **MSE**: Squares errors, penalizes large errors more heavily, sensitive to outliers
- **MAE**: Uses absolute values, treats all errors equally, robust to outliers
- **When to use MSE**: When large errors are particularly bad
- **When to use MAE**: When outliers are expected or all errors are equally important

### 2. "When would you use MAPE vs SMAPE?"

**Answer**:
- **MAPE**: Asymmetric, undefined when actual values are zero, biased toward underforecasting
- **SMAPE**: Symmetric, handles zero values better, bounded between 0-100%
- **Use SMAPE**: When you want symmetric treatment of over/under-forecasting
- **Limitation**: Both are scale-dependent and can be misleading with small denominators

### 3. "How do you interpret R-squared?"

**Answer**:
- **Definition**: Proportion of variance in dependent variable explained by independent variables
- **Range**: 0 to 1 (can be negative for very poor models)
- **Interpretation**: R² = 0.8 means 80% of variance is explained
- **Limitations**: Can be artificially high with many features, doesn't imply causation

### 4. "What is MASE and when is it useful?"

**Answer**:
- **Definition**: Mean Absolute Scaled Error, compares model to naive forecast
- **Formula**: MASE = MAE_model / MAE_naive
- **Interpretation**: <1 means better than naive, >1 means worse than naive
- **Advantage**: Scale-independent, useful for comparing across different datasets

### 5. "How do outliers affect different error metrics?"

**Answer**:
- **MSE/RMSE**: Highly sensitive (quadratic penalty)
- **MAE**: Less sensitive (linear penalty)
- **MAPE**: Can be very sensitive if outliers have small actual values
- **R²**: Can be significantly affected by outliers

## Formulas Reference

### Basic Error Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | (1/n)Σ(y_true - y_pred)² | Lower is better, same units² |
| RMSE | √MSE | Lower is better, same units |
| MAE | (1/n)Σ\|y_true - y_pred\| | Lower is better, same units |
| MAPE | (100/n)Σ\|y_true - y_pred\|/\|y_true\| | Lower is better, percentage |

### Advanced Error Metrics

| Metric | Formula | Range |
|--------|---------|-------|
| SMAPE | (100/n)Σ(2\|y_true - y_pred\|)/(y_true + y_pred) | 0-100% |
| MASE | MAE_model / MAE_naive | 0-∞ |
| MSLE | (1/n)Σ(log(1+y_true) - log(1+y_pred))² | 0-∞ |
| R² | 1 - (SS_res / SS_tot) | -∞ to 1 |

## Error Metrics Comparison

| Metric | Outlier Sensitivity | Scale Dependency | Zero Handling | Best Use Case |
|--------|-------------------|------------------|---------------|---------------|
| MSE | High | Yes | Good | When large errors are costly |
| RMSE | High | Yes | Good | Interpretable version of MSE |
| MAE | Low | Yes | Good | When outliers are expected |
| MAPE | Variable | No | Poor | Percentage interpretation needed |
| SMAPE | Variable | No | Better | Symmetric percentage errors |
| MASE | Low | No | Good | Cross-dataset comparison |
| R² | High | No | Good | Explained variance interpretation |

## Practice Problems

The module includes 6+ HackerRank-style problems covering:

1. **MSE Calculation**: Basic mean squared error implementation
2. **RMSE Calculation**: Root mean squared error implementation
3. **MAE Calculation**: Mean absolute error implementation
4. **MAPE Calculation**: Handling zero values in percentage errors
5. **R-squared Calculation**: Coefficient of determination
6. **Model Comparison**: Comparing models using error metrics

## Metric Selection Guide

### For Regression Problems:
- **General purpose**: RMSE (interpretable, penalizes large errors)
- **With outliers**: MAE (robust to outliers)
- **Percentage interpretation**: MAPE or SMAPE
- **Cross-dataset comparison**: MASE
- **Explained variance**: R²

### For Time Series:
- **Point forecasts**: MASE (scale-independent)
- **Directional accuracy**: MDA
- **Forecast evaluation**: Theil's U

### For Business Applications:
- **Cost-sensitive**: MSE (if large errors are expensive)
- **Reporting**: MAPE (easy to explain to stakeholders)
- **Model selection**: Multiple metrics for comprehensive evaluation

## Running Tests

```bash
# Run all tests
python test_error_metrics.py

# Run specific test class
python -m unittest test_error_metrics.TestErrorMetricsCalculator

# Run with verbose output
python test_error_metrics.py -v
```

## Visualization Examples

The visualization module provides:

1. **Error Analysis**: Actual vs predicted, residuals, error distribution
2. **Model Comparison**: Side-by-side metric comparison
3. **Sensitivity Analysis**: How metrics respond to outliers
4. **Percentage Metrics**: Scale-independent comparisons
5. **Residual Analysis**: Detailed residual diagnostics

## Tips for Interviews

### Conceptual Understanding
1. **Know when to use each metric** (outliers, scale, interpretation)
2. **Understand metric limitations** (MAPE with zeros, R² with many features)
3. **Explain business implications** (cost of different error types)
4. **Discuss metric combinations** (use multiple metrics for complete picture)

### Implementation Tips
1. **Handle edge cases** (zeros, empty arrays, identical values)
2. **Use appropriate data types** (avoid integer overflow)
3. **Validate inputs** (same length arrays, non-negative for MSLE)
4. **Consider computational efficiency** for large datasets

### Common Mistakes to Avoid
1. Using only one metric for model evaluation
2. Not considering the business context when choosing metrics
3. Ignoring outliers when using MSE/RMSE
4. Misinterpreting R² as causation
5. Using MAPE with data containing zeros

## Advanced Topics

For senior positions, be prepared to discuss:

1. **Custom Loss Functions**: Designing metrics for specific business needs
2. **Probabilistic Metrics**: Log-likelihood, Brier score for probabilistic predictions
3. **Multi-output Metrics**: Handling multiple target variables
4. **Time Series Metrics**: Seasonal adjustments, forecast horizons
5. **Robust Metrics**: Huber loss, quantile regression metrics
6. **Bayesian Metrics**: Posterior predictive checks, information criteria

## Real-World Applications

### Finance
- **Portfolio optimization**: Tracking error, information ratio
- **Risk modeling**: Value at Risk, Expected Shortfall
- **Algorithmic trading**: Sharpe ratio, maximum drawdown

### Healthcare
- **Clinical trials**: Sensitivity, specificity, AUC
- **Diagnostic models**: Precision, recall, F1-score
- **Treatment effectiveness**: Number needed to treat

### E-commerce
- **Demand forecasting**: MAPE for inventory planning
- **Recommendation systems**: Precision@K, NDCG
- **A/B testing**: Statistical significance, effect size

### Manufacturing
- **Quality control**: Control charts, process capability indices
- **Predictive maintenance**: Remaining useful life accuracy
- **Supply chain**: Forecast bias, forecast accuracy

## Resources for Further Learning

1. **Books**:
   - "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos
   - "The Elements of Statistical Learning" by Hastie et al.

2. **Online Courses**:
   - Coursera: Machine Learning courses
   - edX: Statistics and Data Science programs

3. **Practice Platforms**:
   - Kaggle: Competition metrics and evaluation
   - HackerRank: Statistics domain problems
   - LeetCode: Algorithm and data structure problems

## Contributing

To add new error metrics or problems:

1. Add the method to `ErrorMetricsCalculator` class
2. Create corresponding tests in `test_error_metrics.py`
3. Add visualization if applicable
4. Update this documentation

## License

This educational material is provided under the MIT License for interview preparation purposes.
