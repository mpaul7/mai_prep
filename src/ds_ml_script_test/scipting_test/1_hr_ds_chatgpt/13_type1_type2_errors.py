"""_summary_
Concept
========
Type I and Type II errors are fundamental concepts in statistical hypothesis testing.

Type I Error (False Positive):
- Rejecting the null hypothesis when it is actually true
- α (alpha) = probability of Type I error
- Example: Convicting an innocent person

Type II Error (False Negative): 
- Failing to reject the null hypothesis when it is actually false
- β (beta) = probability of Type II error
- Example: Letting a guilty person go free

Relationship Table:
| Reality vs Decision | Null True (H₀)     | Null False (H₁)    |
|--------------------|--------------------|--------------------|
| Reject Null        | Type I Error (α)   | Correct Decision   |
| Accept Null        | Correct Decision   | Type II Error (β)  |

Key Points:
1. Trade-off between Type I and II errors
2. Decreasing one type typically increases the other
3. Power = 1 - β = probability of correctly rejecting false null
4. α is typically set at 0.05 (5% significance level)

"""
import numpy as np
from scipy import stats

import numpy as np
from scipy import stats

# ------------------------------
# HackerRank-style input (hardcoded)
# ------------------------------
n_trials = 1000      # Number of hypothesis tests to run
n_samples = 30       # Sample size per test
null_mean = 100      # Population mean under H0
alt_mean = 102       # True mean under alternative H1
std = 15             # Population standard deviation
alpha = 0.05         # Significance level

# ------------------------------
# Simulate tests under H0 (Type I error)
# ------------------------------
null_samples = np.random.normal(null_mean, std, (n_trials, n_samples))
null_t_stats = (np.mean(null_samples, axis=1) - null_mean) / (std / np.sqrt(n_samples))
null_p_values = 2 * (1 - stats.t.cdf(np.abs(null_t_stats), n_samples-1))
type_1_errors = np.sum(null_p_values < alpha)
type_1_rate = type_1_errors / n_trials

# ------------------------------
# Simulate tests under H1 (Type II error)
# ------------------------------
alt_samples = np.random.normal(alt_mean, std, (n_trials, n_samples))
alt_t_stats = (np.mean(alt_samples, axis=1) - null_mean) / (std / np.sqrt(n_samples))
alt_p_values = 2 * (1 - stats.t.cdf(np.abs(alt_t_stats), n_samples-1))
type_2_errors = np.sum(alt_p_values >= alpha)
type_2_rate = type_2_errors / n_trials
power = 1 - type_2_rate

# ------------------------------
# Print Results
# ------------------------------
print("\nHypothesis Testing Simulation Results:")
print("-" * 50)
print(f"Type I Error Rate (α): {type_1_rate:.3f}")
print(f"Type II Error Rate (β): {type_2_rate:.3f}")
print(f"Statistical Power (1-β): {power:.3f}")

print("\nInterpretation:")
print("-" * 50)
print(f"• {type_1_rate*100:.1f}% of the time we incorrectly rejected a true null hypothesis")
print(f"• {type_2_rate*100:.1f}% of the time we failed to reject a false null hypothesis")
print(f"• The test has {power*100:.1f}% power to detect the specified effect size")

