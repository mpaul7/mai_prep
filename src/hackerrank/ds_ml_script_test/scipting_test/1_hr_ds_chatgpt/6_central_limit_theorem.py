"""_summary_
Concept: The Central Limit Theorem states:

The sampling distribution of the sample mean will be approximately normal, regardless of the population’s distribution, 
if the sample size is sufficiently large (usually n ≥ 30).

The theorem is powerful because it allows us to make inferences about the population mean even when we don’t know the population’s distribution.


Key Points:

1. The mean of the sampling distribution = population mean (μ)
2. The standard deviation of the sampling distribution (standard error) = σ/√n
3. Larger samples → sampling distribution more closely approximates a normal distribution

Why CLT is Important
=====================
1. Allows us to use normal distribution assumptions for hypothesis testing and confidence intervals.

2. Works even if population is not normal.

3. Forms the basis for z-tests, t-tests, and many ML algorithms.

Formulats: 
=========
1. Standard Error = σ/√n
2. Z-score = (x̄ - μ) / (σ/√n)
3. t-score = (x̄ - μ) / (s/√n)
4. Confidence Interval = x̄ ± Z*σ/√n
5. Hypothesis Test = t-test or z-test
6. ML Algorithms = Linear Regression, Logistic Regression, Random Forest, etc.
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_clt(population_size=10000, sample_size=50, num_samples=1000):
    # Generate a non-normal population (e.g., exponential)
    population = np.random.exponential(scale=2.0, size=population_size)
    pop_mean = np.mean(population)
    pop_std = np.std(population)
    
    # Optional: plot histogram of population
    plt.hist(population, bins=30, edgecolor='black')
    plt.title("Population Distribution (Exponential)")
    plt.xlabel("Population")
    plt.ylabel("Frequency")
    plt.show()
    
    print(f"Population Mean: {pop_mean:.2f}, Population SD: {pop_std:.2f}")
    
    # Collect sample means
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_means.append(np.mean(sample))
    
    sample_means = np.array(sample_means)
    mean_of_sample_means = np.mean(sample_means)
    se = np.std(sample_means)
    
    print(f"Mean of Sample Means: {mean_of_sample_means:.2f}")
    print(f"Standard Error (SE): {se:.2f}")
    
    # Optional: plot histogram of sample means
    plt.hist(sample_means, bins=30, edgecolor='black')
    plt.title("Sampling Distribution of the Mean (CLT)")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")
    plt.show()
def hackerrank_style_problems():
    """
    Problem:
    =========================================================
    1. Generate a population of 10,000 numbers using Poisson distribution (λ=3).
    2. Take 50 random samples 1,000 times.
    3. Compute mean and standard error of sample means.
    """
    print(f"\n\nProblem: Generate a population of 10,000 numbers using Poisson distribution (λ=3).")
    print(f"Take 50 random samples 1,000 times.")
    print(f"Compute the mean and standard error of the sampling distribution of the mean.")
    import numpy as np

    population = np.random.poisson(lam=3, size=10000)
    sample_means = []
    
    # Optional: plot histogram of sample means
    plt.hist(population, bins=30, edgecolor='black')
    plt.title("Population Distribution (Poisson)")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")
    plt.show()

    for _ in range(1000):
        sample = np.random.choice(population, size=50, replace=False)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)
    print("Mean of Sample Means:", round(np.mean(sample_means),2))
    print("Standard Error:", round(np.std(sample_means),2))

    # Optional: plot histogram of sample means
    plt.hist(sample_means, bins=30, edgecolor='black')
    plt.title("Sampling Distribution of the Mean (CLT)")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    simulate_clt()
    hackerrank_style_problems()


"""_summary_
Observation:
============
1. Population is exponential (skewed) → not normal  
2. Sample size = 50 → sampling distribution of the mean → approximately normal (bell-shaped)
3. SE decreases as sample size increases
"""

"""
 
"""