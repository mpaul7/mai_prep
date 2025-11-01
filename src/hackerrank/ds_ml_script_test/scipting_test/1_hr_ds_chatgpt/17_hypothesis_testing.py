"""_summary_
A statistical hypothesis test helps us decide whether to accept or reject a null hypothesis H₀
about a population parameter (mean, variance, proportion, etc.) using sample data.

Core Concepts:
-------------
| Term                        | Meaning                                                      |
|---------------------------- |--------------------------------------------------------------|
| Null Hypothesis (H₀)        | The default assumption — no effect, no difference            |
| Alternative Hypothesis (H₁) | The claim you want to prove — an effect or difference exists |
| Test Statistic              | A standardized value (z, t, or χ²) computed from data        |
| p-value                     | Probability of getting a result as extreme as observed,      |
|                             | if H₀ were true                                              |
| α (Significance Level)      | Threshold to reject H₀, usually 0.05 (5%)                    |
| Decision Rule               | If p-value < α → reject H₀; otherwise fail to reject H₀      |

Common Hypothesis Tests
----------------------
Test                | Use Case                                  | Test Statistic
--------------------|-------------------------------------------|---------------
Z-test             | Known variance or large sample (n > 30)    | z = (x̄ - μ)/(σ/√n)
t-test (1-sample)  | Unknown variance, small sample            | t = (x̄ - μ)/(s/√n)
t-test (2-sample)  | Compare two sample means                  | t = (x̄₁ - x̄₂)/√(s₁²/n₁ + s₂²/n₂)
Chi-square test    | For categorical/discrete data             | Tests variance or independence
"""

import numpy as np
from scipy import stats

# ------------------------------
# 1️⃣ One-sample t-test
# ------------------------------

"""_summary_
Problem Statement

You are given a sample of numeric data and a population mean (μ₀). 
Perform a one-sample t-test to determine if the sample mean differs significantly from the population mean 
at a significance level α = 0.05.
compute t-statistic and p-value.
compare p-value with α to decide whether to reject H₀.
t = (x̄ - μ)/(s/√n)
"""
n_values = "8"
data_values = "12 11 13 15 14 10 13 14"
mu_0 = "12"

n_values = int(n_values.strip())
data = np.array(list(map(float, data_values.split())))
mu_0 = float(mu_0.strip())

# ------------------------------
# Compute sample statistics
# ------------------------------
# x_bar = np.mean(data)
# s = np.std(data, ddof=1)
# n = len(data)
# t_stat = (x_bar - mu0) / (s / np.sqrt(n))
# ------------------------------
# Compute t-statistic and p-value
# ------------------------------
t_stat, p_val = stats.ttest_1samp(data, mu_0)
print(f"t_stat = {t_stat:.3f}, p_val = {p_val:.4f}")
print("---- One-Sample t-test ----")
print(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
if p_val < 0.05:
    print("Reject H₀ → sample mean differs significantly from population mean\n")
else:
    print("Fail to reject H₀ → no significant difference\n")

# ------------------------------
# 2️⃣ Two-sample t-test (independent)
# ------------------------------
group_A = np.array([23, 21, 25, 20, 22])
group_B = np.array([30, 28, 27, 29, 31])

t_stat2, p_val2 = stats.ttest_ind(group_A, group_B)
print("---- Two-Sample t-test ----")
print(f"t-statistic = {t_stat2:.3f}, p-value = {p_val2:.4f}")
if p_val2 < 0.05:
    print("Reject H₀ → groups differ significantly\n")
else:
    print("Fail to reject H₀ → no significant difference\n")

# ------------------------------
# 3️⃣ Z-test (manual, large sample)
# ------------------------------
sample_mean = 52
pop_mean = 50
pop_std = 5
n = 36
z_stat = (sample_mean - pop_mean) / (pop_std / np.sqrt(n))
p_val_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
print("---- Z-test ----")
print(f"z-statistic = {z_stat:.3f}, p-value = {p_val_z:.4f}")
if p_val_z < 0.05:
    print("Reject H₀ → sample mean differs from population mean\n")
else:
    print("Fail to reject H₀\n")

# ------------------------------
# 4️⃣ Chi-square test (independence)
# ------------------------------
observed = np.array([[20, 30], [15, 35]])  # contingency table
chi2, p_chi, dof, expected = stats.chi2_contingency(observed)
print("---- Chi-square Test ----")
print(f"Chi2 = {chi2:.3f}, p-value = {p_chi:.4f}")
print("Expected frequencies:\n", np.round(expected, 2))
if p_chi < 0.05:
    print("Reject H₀ → variables are dependent\n")
else:
    print("Fail to reject H₀ → variables are independent\n")
