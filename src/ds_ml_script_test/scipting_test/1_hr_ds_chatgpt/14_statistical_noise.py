"""Statistical Noise in Data Science

Concept
=======
Statistical noise is random variation in data that makes it difficult to see the underlying pattern,
cannot be explained by the model, or is not related to the independent variable.

Example in regression:
    Y = m(X|θ) + ε

where:
    Y = dependent variable
    X = independent variable
    m = model
    θ = model parameters
    ε = error (noise)

If ε is high, the model is not able to explain the variation in Y.

Key Points
==========
1. Noise is unpredictable; part of data we cannot model.
2. Reducing noise is impossible, but we can minimize its effect using good models and larger samples.
3. Variance in predictions often comes from noise.

Intuition
=========
1. Imagine measuring heights of people with a slightly inaccurate ruler.
2. True heights: [170, 165, 180]
3. Measured heights: [171, 164, 181]
4. Differences (+1, -1, +1) → statistical noise

Common Sources of Statistical Noise
===================================
1. Measurement error
2. Sampling error
3. Random variation
4. Model error
5. Systematic error
"""

import numpy as np

np.random.seed(42)

# Generate X values
n = 50
X = np.linspace(0, 10, n)

# True model parameters
m = 2.5  # slope
c = 1.0  # intercept

# Add noise (ε ~ N(0, σ²))
sigma = 3  # standard deviation of noise
epsilon = np.random.normal(0, sigma, n)  # noise vector

# Generate Y with noise
Y = m * X + c + epsilon

# Output first 10 points
print("X values:", np.round(X[:10], 2).tolist())
print("Y values with noise:", np.round(Y[:10], 2).tolist())
print("Noise (epsilon) samples:", np.round(epsilon[:10], 2).tolist())

"""
Interpretation
==============
Y = mX + c → underlying trend (signal)
ε → random noise added to each observation

Noise causes scatter around the regression line.

In practice:
- Increasing sample size reduces effect of noise on model estimates
- Using robust models can reduce variance from noise

HackerRank Notes
==============

Noise is always modeled as random variable with mean 0.

Simulation:
- Use np.random.normal(0, sigma, n) for Gaussian noise
- Helps practice regression with realistic data

Key Observation:
- Predicted line (mX + c) differs from actual Y values due to noise term ε
"""

"""_summary_
HackerRank-Style Questions on Statistical Noise

1. Conceptual Multiple Choice Questions
----------------------------------------

Q1. What does statistical noise in a regression model represent?
    A) The predicted value of Y
    B) Variability in data unexplained by the model ✓ 
    C) The slope of the regression line
    D) The correlation coefficient

Q2. In the model Y = mX + c + ε, what is ε?
    A) Slope of the line
    B) Noise or error term ✓
    C) Intercept 
    D) Mean of Y

Q3. Which statement about noise is true?
    A) Noise can always be eliminated
    B) Noise has mean zero in classical regression assumptions ✓
    C) Noise increases the slope of the line
    D) Noise is deterministic

2. Coding/Simulation Questions
------------------------------

Common HackerRank Tasks:

1. Generate Dataset with Noise
   - Input: X values and parameters for linear model
   - Task: Add Gaussian noise to Y

2. Fit Regression Line to Noisy Data
   - Input: X and noisy Y values
   - Task: Compute slope, intercept, predicted Y

3. Compute Error Metrics
   - Input: X and noisy Y values
   - Task: Calculate MSE, RMSE, MASE for fitted line

4. Analyze Noise Statistics
   - Input: Noisy dataset
   - Task: Report mean and standard deviation of noise (ε)
"""