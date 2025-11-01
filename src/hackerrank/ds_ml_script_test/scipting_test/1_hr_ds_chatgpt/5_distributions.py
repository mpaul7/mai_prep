"""_summary_
Concept : Distribution = Pattern of how data is distributed
Distributions can be discrete (specific values) or continuous (any value in a range).
distributions can be discrete or continuous.
discrete distributions = specific values, example: number of heads in 10 coin flips
continuous distributions = any value in a range, example: height of a person

Common Distributions
| Distribution      | Type                | Description                               | NumPy Example                               |
| ---------------- | -------------------  | -------------------------------------    | ------------------------------------------ |
| Bernoulli        | Discrete             | Two outcomes: success (1) or failure (0) | np.random.binomial(n=1, p=0.3, size=10)    |
| Uniform          | Discrete / Continuous| All outcomes equally likely              | np.random.uniform(low=0, high=1, size=10)  |
| Binomial         | Discrete             | Number of successes in n trials          | np.random.binomial(n=10, p=0.5, size=10)   |
| Normal (Gaussian)| Continuous           | Bell-shaped curve; mean μ, std σ         | np.random.normal(loc=0, scale=1, size=10)  |
| Poisson          | Discrete             | Number of events in fixed interval       | np.random.poisson(lam=3, size=10)          |
| Exponential      | Continuous           | Time between events                      | np.random.exponential(scale=1.0, size=10)  |


Common Measures of Central Tendency
| Measure           | Formula                           | Description                           |
| ----------------- | --------------------------------- | ------------------------------------- |
| Mean              | Σx/n                              | Average value                         |
| Weighted Mean     | Σ(x_i * w_i) / Σw_i              | Average value with weights            |
| Median            | Middle value of sorted data       | Middle value of sorted data          |
| Mode              | Most frequent value               | Value that appears most often         |
| Geometric Mean    | (x_1 * x_2 * ... * x_n)^(1/n)    | Average rate of growth               |
| Harmonic Mean     | n / Σ(1/x_i)                     | Inverse of average of inverses        |
| Trimmed Mean      | Average of middle 80% of data    | Average of middle 80% of data        |
| Interquartile Mean| Average of Q1 and Q3             | Average of 25th and 75th percentiles  |


Notes & HackerRank Relevance
==============================  
1. Bernoulli → Often used for coin flips, binary outcomes

2. Binomial → Number of successes in repeated trials

3. Uniform → All outcomes equally likely, random sampling

4. Normal → Central limit theorem, z-scores, many ML assumptions

5. Poisson → Count of events in fixed interval, e.g., calls/hour

6. Exponential → Time between events, reliability modeling

HackerRank tasks often require:
===============================
1. Generate random samples from a distribution

2. Compute mean, variance, probability

3. Simulate experiments (e.g., coin tosses, dice rolls)

"""

import numpy as np

def generate_distributions():
    print("=== Bernoulli Distribution (p=0.3, n=1) ===")
    bern = np.random.binomial(n=1, p=0.3, size=10)
    print(bern)

    print("\n=== Uniform Distribution [0,1) ===")
    uniform = np.random.uniform(0, 1, size=10)
    print(np.round(uniform, 2))

    print("\n=== Binomial Distribution (n=10, p=0.5) ===")
    binom = np.random.binomial(n=10, p=0.5, size=10)
    print(binom)

    print("\n=== Normal Distribution (μ=0, σ=1) ===")
    normal = np.random.normal(loc=0, scale=1, size=10)
    print(np.round(normal, 2))

    print("\n=== Poisson Distribution (λ=3) ===")
    poisson = np.random.poisson(lam=3, size=10)
    print(poisson)

    print("\n=== Exponential Distribution (scale=1.0) ===")
    exp = np.random.exponential(scale=1.0, size=10)
    print(np.round(exp, 2))

def hackerrank_style_problems():
    """
    Collection of HackerRank-style problems for distributions.
    =========================================================
    Problem: Simulate 10 coin flips (Bernoulli with p=0.3) and 10 dice rolls (Uniform discrete 1-6).
    Compute mean and variance for both.
    """
    import numpy as np

    print(f"\n\nProblem: Simulate 10 coin flips (Bernoulli with p=0.3) and 10 dice rolls (Uniform discrete 1-6).")
    print(f"Compute mean and variance for both.")
    # Bernoulli coin flips
    coin_flips = np.random.binomial(n=1, p=0.3, size=10)
    print(f"Coin flips:", coin_flips)
    print(f"Mean:", round(np.mean(coin_flips),2), "Variance:", round(np.var(coin_flips),2))

    # Uniform dice rolls
    dice = np.random.randint(1,7,size=10)
    print(f"Dice rolls:", dice)
    print(f"Mean:", round(np.mean(dice),2), "Variance:", round(np.var(dice),2))

    


if __name__ == "__main__":
    generate_distributions()
    hackerrank_style_problems()
