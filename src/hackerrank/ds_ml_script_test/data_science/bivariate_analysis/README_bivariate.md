# Bivariate Analysis

This module provides comprehensive tools for bivariate analysis, specifically designed for data science interview preparation.

## Overview

Bivariate analysis examines the relationship between two variables. Understanding correlation, covariance, regression, and goodness of fit is crucial for data scientists and is frequently tested in technical interviews.

## Key Concepts Covered

### 1. Correlation Analysis
- **Pearson Correlation**: Linear relationship strength (-1 to +1)
- **Spearman Correlation**: Monotonic relationship (rank-based)
- **Kendall's Tau**: Alternative rank correlation (optional)

### 2. Covariance
- **Sample Covariance**: Σ(x-x̄)(y-ȳ)/(n-1)
- **Population Covariance**: Σ(x-x̄)(y-ȳ)/n
- **Interpretation**: Direction of linear relationship

### 3. Linear Regression
- **Simple Linear Regression**: y = mx + b
- **Least Squares Method**: Minimize sum of squared residuals
- **Regression Coefficients**: Slope and intercept calculation

### 4. Goodness of Fit
- **R-squared**: Proportion of variance explained
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

## Files Structure

```
src/bivariate_analysis/
├── bivariate.py              # Main bivariate analysis toolkit
├── test_bivariate.py         # Comprehensive test suite
├── bivariate_visualization.py # Visualization tools
└── README_bivariate.md       # This documentation
```

## Quick Start

### Basic Usage

```python
from bivariate import BivariateAnalyzer

# Initialize analyzer
analyzer = BivariateAnalyzer()

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Calculate Pearson correlation
corr = analyzer.calculate_pearson_correlation(x, y)
print(f"Correlation: {corr['correlation']:.3f}")  # 1.000

# Perform linear regression
regression = analyzer.simple_linear_regression(x, y)
print(f"Equation: {regression['equation']}")  # y = 2.0000x + 0.0000

# Calculate goodness of fit
gof = analyzer.calculate_goodness_of_fit(y, regression['predictions'])
print(f"R-squared: {gof['r_squared']:.3f}")  # 1.000
```

### HackerRank-Style Problems

```python
from bivariate import hackerrank_bivariate_problems

# Run practice problems
hackerrank_bivariate_problems()
```

### Visualization

```python
from bivariate_visualization import BivariateVisualizer

visualizer = BivariateVisualizer()
visualizer.plot_correlation_analysis(x, y, "Correlation Analysis")
```

## Common Interview Questions

### 1. "What's the difference between correlation and covariance?"

**Answer**:
- **Covariance**: Measures direction of linear relationship (units: x_units × y_units)
- **Correlation**: Standardized covariance, measures strength and direction (-1 to +1)
- **Formula**: r = Cov(X,Y) / (σₓ × σᵧ)
- **Advantage of correlation**: Scale-independent, easier to interpret

### 2. "When would you use Spearman vs Pearson correlation?"

**Answer**:
- **Pearson**: Linear relationships, normally distributed data, no outliers
- **Spearman**: Monotonic relationships, ordinal data, presence of outliers
- **Example**: Pearson for height vs weight; Spearman for education level vs income rank

### 3. "How do you interpret R-squared?"

**Answer**:
- **Definition**: Proportion of variance in Y explained by X
- **Range**: 0 to 1 (0% to 100% of variance explained)
- **Interpretation**: 
  - R² = 0.7 means 70% of variance is explained
  - Remaining 30% is unexplained (residual variance)
- **Caution**: High R² doesn't imply causation

### 4. "What assumptions does linear regression make?"

**Answer**:
1. **Linearity**: Relationship between X and Y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: (for multiple regression)

### 5. "How do outliers affect correlation and regression?"

**Answer**:
- **Correlation**: Can artificially inflate or deflate correlation
- **Regression**: Can dramatically change slope and intercept
- **Solutions**: Use robust methods (Spearman), remove outliers, or use robust regression
- **Detection**: Residual plots, leverage plots, Cook's distance

## Formulas Reference

### Correlation Formulas

| Type | Formula |
|------|---------|
| Pearson | r = Σ(x-x̄)(y-ȳ) / √[Σ(x-x̄)² × Σ(y-ȳ)²] |
| Spearman | ρ = 1 - (6Σd²) / (n(n²-1)) |
| Covariance | Cov(X,Y) = Σ(x-x̄)(y-ȳ) / (n-1) |

### Regression Formulas

| Parameter | Formula |
|-----------|---------|
| Slope (m) | m = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)² |
| Intercept (b) | b = ȳ - m×x̄ |
| R-squared | R² = 1 - (SS_res / SS_tot) |

### Goodness of Fit Formulas

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | √(Σ(y-ŷ)²/n) | Average prediction error |
| MAE | Σ\|y-ŷ\|/n | Average absolute error |
| MAPE | (100/n)Σ\|y-ŷ\|/y | Average percentage error |

## Practice Problems

The module includes 5+ HackerRank-style problems covering:

1. **Pearson Correlation Calculation**: Manual implementation
2. **Covariance Calculation**: Sample vs population
3. **Linear Regression**: Slope and intercept calculation
4. **R-squared Calculation**: Goodness of fit measure
5. **Correlation Strength Classification**: Interpreting correlation values

## Correlation Interpretation Guide

| |r| Range | Strength | Interpretation |
|-----------|----------|----------------|
| 0.9 - 1.0 | Very Strong | Almost perfect relationship |
| 0.7 - 0.9 | Strong | Strong relationship |
| 0.5 - 0.7 | Moderate | Moderate relationship |
| 0.3 - 0.5 | Weak | Weak relationship |
| 0.0 - 0.3 | Very Weak | Little to no relationship |

## R-squared Interpretation Guide

| R² Range | Interpretation | Model Quality |
|----------|----------------|---------------|
| 0.9 - 1.0 | Excellent fit | Very high explanatory power |
| 0.7 - 0.9 | Good fit | High explanatory power |
| 0.5 - 0.7 | Moderate fit | Moderate explanatory power |
| 0.3 - 0.5 | Poor fit | Low explanatory power |
| 0.0 - 0.3 | Very poor fit | Very low explanatory power |

## Running Tests

```bash
# Run all tests
python test_bivariate.py

# Run specific test class
python -m unittest test_bivariate.TestBivariateAnalyzer

# Run with verbose output
python test_bivariate.py -v
```

## Visualization Examples

The visualization module provides:

1. **Correlation Analysis**: Scatter plots with regression lines
2. **Regression Diagnostics**: Residual plots, Q-Q plots
3. **Correlation Matrix**: Heatmaps for multiple variables
4. **Relationship Types**: Different correlation patterns
5. **Goodness of Fit Comparison**: Multiple model comparison

## Tips for Interviews

### Conceptual Understanding
1. **Distinguish correlation from causation**
2. **Know when to use different correlation types**
3. **Understand regression assumptions**
4. **Interpret goodness of fit metrics correctly**

### Implementation Tips
1. **Handle edge cases** (identical values, outliers)
2. **Use appropriate formulas** (sample vs population)
3. **Validate assumptions** before applying methods
4. **Consider robust alternatives** when assumptions are violated

### Common Mistakes to Avoid
1. Confusing correlation with causation
2. Using Pearson correlation for non-linear relationships
3. Ignoring outliers in regression analysis
4. Over-interpreting R-squared values
5. Not checking regression assumptions

## Advanced Topics

For senior positions, be prepared to discuss:

1. **Multiple Regression**: Multiple predictors, adjusted R²
2. **Non-linear Regression**: Polynomial, exponential models
3. **Robust Regression**: Handling outliers and violations
4. **Regularized Regression**: Ridge, Lasso, Elastic Net
5. **Regression Diagnostics**: Leverage, influence, Cook's distance
6. **Bootstrap Confidence Intervals**: For correlation and regression parameters

## Real-World Applications

### Business Analytics
- **Sales forecasting**: Revenue vs advertising spend
- **Customer analysis**: Satisfaction vs loyalty scores
- **Market research**: Price vs demand relationships

### Finance
- **Portfolio analysis**: Asset correlations
- **Risk modeling**: Beta calculation (stock vs market)
- **Economic indicators**: GDP vs unemployment correlation

### Healthcare
- **Clinical trials**: Treatment dose vs response
- **Epidemiology**: Risk factor correlations
- **Biomarker analysis**: Protein levels vs disease progression

### Machine Learning
- **Feature selection**: Correlation with target variable
- **Model evaluation**: Predicted vs actual correlations
- **Data preprocessing**: Multicollinearity detection

## Resources for Further Learning

1. **Books**:
   - "Applied Linear Statistical Models" by Kutner et al.
   - "Introduction to Statistical Learning" by James et al.

2. **Online Courses**:
   - Coursera: Regression Models
   - edX: Introduction to Linear Models

3. **Practice Platforms**:
   - HackerRank: Statistics domain
   - Kaggle Learn: Intermediate Machine Learning
   - DataCamp: Correlation and Regression courses

## Contributing

To add new bivariate analysis methods or problems:

1. Add the method to `BivariateAnalyzer` class
2. Create corresponding tests in `test_bivariate.py`
3. Add visualization if applicable
4. Update this documentation

## License

This educational material is provided under the MIT License for interview preparation purposes.
