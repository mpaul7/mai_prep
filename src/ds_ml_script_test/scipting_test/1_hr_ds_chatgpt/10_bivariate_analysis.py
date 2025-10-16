"""_summary_

Concept
========
Bivariate analysis examines the relationship between two variables.

Common Methods of Bivariate Analysis:
====================================
| Method               | Formula                                     | Description                               |
|----------------------|---------------------------------------------|------------------------------------------ |
| Pearson Correlation  | r = Σ(x-x̄)(y-ȳ) / √[Σ(x-x̄)² × Σ(y-ȳ)²]      | Linear relationship strength (-1 to +1)   |
| Spearman Correlation | ρ = 1 - (6Σd²) / (n(n²-1))                  | Monotonic relationship (rank-based)       |
| Covariance           | Cov(X,Y) = Σ(x-x̄)(y-ȳ) / (n-1)              | Direction of linear relationship          |
| Linear Regression    | y = mx + b                                  | Relationship between two variables        |
| Goodness of Fit      | R² = 1 - (SS_res / SS_tot)                  | Proportion of variance explained          |

Notes & HackerRank Relevance:
============================
1. Pearson Correlation → Linear relationship strength (-1 to +1)
2. Spearman Correlation → Monotonic relationship (rank-based)
3. Covariance → Direction of linear relationship
4. Correlation Measures strength and direction (scaled -1 to +1)
5. Linear Regression → Relationship between two variables
6. Goodness of Fit → Proportion of variance explained
"""

import numpy as np

def bivariate_analysis(x, y):
    """
    Perform bivariate analysis:
    - Covariance
    - Correlation
    - Least-squares regression (slope, intercept)
    - Goodness-of-fit (R²)
    
    Key HackerRank Notes:
    - Check lengths of X and Y
    - Use NumPy vectorized operations for performance
    - Include R² for evaluating fit
    """
    
    # Check lengths
    if len(x) != len(y):
        raise ValueError("X and Y must have the same length")
    
    x = np.array(x)
    y = np.array(y)
    
    # Covariance & Correlation
    cov = np.cov(x, y, ddof=1)[0,1]           # Sample covariance
    corr = np.corrcoef(x, y)[0,1]            # Pearson correlation
    
    # Least Squares Regression
    x_mean, y_mean = np.mean(x), np.mean(y)
    m = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)  # Slope
    c = y_mean - m*x_mean                                           # Intercept
    
    y_pred = m*x + c                                                # Predicted values
    
    # Goodness-of-fit R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r2 = 1 - ss_res/ss_tot
    
    return round(cov,2), round(corr,2), round(m,2), round(c,2), round(r2,2)

if __name__ == "__main__":
    X_values = "1 2 3 4 5"
    Y_values = "2 4 5 4 5"
    # Input: space-separated numbers
    x = list(map(float, X_values.split()))
    y = list(map(float, Y_values.split()))
    
    cov, corr, slope, intercept, r2 = bivariate_analysis(x, y)
    
    print(f"Covariance: {cov} - Covariance positive → X and Y increase together")
    print(f"Correlation: {corr} - Correlation positive → X and Y increase together")
    print(f"Regression Line: Y = {slope}X + {intercept}")
    print(f"R²: {r2} - [R² = 1 → Perfect fit] 0.71 → 71% of Y’s variance explained by X")
