"""_summary_
| Metric              | Formula                               | Denominator               |
| ------------------- | ------------------------------------- | ------------------------- |
| Population Variance | ( \frac{\sum(x_i - \mu)^2}{N} )       | N                         |
| Sample Variance     | ( \frac{\sum(x_i - \bar{x})^2}{n-1} ) | n-1 (Bessel’s correction) |

"""

import numpy as np

def population_variance_sd(data):
    """
    Returns population variance and SD
    """
    variance = np.var(data)       # By default, np.var uses ddof=0
    sd = np.sqrt(variance)
    return round(variance, 2), round(sd, 2)

def sample_variance_sd(data):
    """
    Returns sample variance and SD
    """
    variance = np.var(data, ddof=1)  # ddof=1 for sample
    sd = np.sqrt(variance)
    return round(variance, 2), round(sd, 2)

if __name__ == "__main__":
    # Input: space-separated numbers
    data_str = "1 2 3 4 5 6 7 8 9 10"
    data = list(map(float, data_str.split()))
    arr = np.array(data)
    
    pop_var, pop_sd = population_variance_sd(arr)
    samp_var, samp_sd = sample_variance_sd(arr)
    
    print(f"Population Variance: {pop_var:.2f}, Population SD: {pop_sd:.2f}")
    print(f"Sample Variance: {samp_var:.2f}, Sample SD: {samp_sd:.2f}")
