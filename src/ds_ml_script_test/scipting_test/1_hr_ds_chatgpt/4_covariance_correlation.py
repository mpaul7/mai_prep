"""_summary_

Covariance = Measure of how two variables vary in tandem from their means
Correlation = Measure of how two variables vary in tandem from their means
Cov(X, Y) = Σ(x-x̄)(y-ȳ)/(n-1) (sample)
Cov(X, Y) = Σ(x-μ)(y-μ)/n (population)

positive covariance = variables tend to increase/decrease together
negative covariance = variables tend to move in opposite directions
zero covariance = variables are independent

Correlation =  Correlation standardized covariance to -1 to 1

r = Cov(X, Y) / (SD(X) * SD(Y))

where: 
r = 1 = perfect positive correlation
r = -1 = perfect negative correlation
r = 0 = no correlation

Covariance  shows the direction and magnitude of the relationship
Correlation shows the strength and direction of the relationship

"""

import numpy as np

def covariance(x, y, sample=True):
    """
    Calculate covariance between two variables.
    sample=True → divide by n-1
    sample=False → divide by N (population)
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    cov = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    if sample:
        cov /= (n - 1)
    else:
        cov /= n
    return round(cov, 2)

def correlation(x, y):
    """
    Calculate Pearson correlation coefficient
    """
    x = np.array(x)
    y = np.array(y)
    corr = np.cov(x, y, ddof=1)[0,1] / (np.std(x, ddof=1) * np.std(y, ddof=1))
    return round(corr, 2)

if __name__ == "__main__":
    # Input: space-separated numbers
    x_str = "1 2 3 4 5"
    y_str = "2 4 5 4 5"
    x = list(map(float, x_str.split()))
    y = list(map(float, y_str.split()))
    
    cov_sample = covariance(x, y, sample=True)
    cov_population = covariance(x, y, sample=False)
    corr = correlation(x, y)
    
    print(f"Sample Covariance: {cov_sample}")
    print(f"Population Covariance: {cov_population}")
    print(f"Correlation: {corr}")
