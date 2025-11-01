# Two sample t-test example
sample1 = np.array([2.3, 1.9, 2.8, 2.5, 2.7])
sample2 = np.array([3.1, 2.9, 3.2, 3.0, 3.3])

# Means and std
mean1 = np.mean(sample1)
mean2 = np.mean(sample2)
print("mean1:", mean1)
print("mean2:", mean2)
std1 = np.std(sample1, ddof=1)
std2 = np.std(sample2, ddof=1)
print("std1:", std1)
print("std2:", std2)
n1 = len(sample1)
n2 = len(sample2)

# t-statistic (independent samples, equal variance not assumed)
t_stat = (mean1 - mean2) / np.sqrt(std1**2/n1 + std2**2/n2)
print("t-statistic:", t_stat)

# Degrees of freedom (Welch-Satterthwaite)
df = ((std1**2/n1 + std2**2/n2)**2) / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
print("Degrees of freedom:", df)

# p-value (two-tailed)
from scipy.stats import t
p_value = 2 * (1 - t.cdf(abs(t_stat), df))
print("p-value:", p_value)
