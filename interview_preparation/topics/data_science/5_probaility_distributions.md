

# Probability Distributions in Machine Learning

Probability distributions are fundamental in data science and machine learning—they help us model uncertainty, describe data, and choose appropriate algorithms for different tasks. Let's break down the most important distributions, their formulas, Python usage, and real-world business impact.

***
## Types of Probability Distributions

### 1 **Normal (Gaussian) Distribution**

- **Definition:** Models continuous data that cluster around a mean (bell-shaped curve).
- **Formula:**

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Where \$\mu\$ is the mean, \$\sigma\$ is the standard deviation.
- **Business Impact:** Used for modeling errors/residuals in regression, heights, test scores, and many natural phenomena.
- **Python Example:**

```python
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-3, 3, 100)
plt.plot(x, norm.pdf(x, loc=0, scale=1), label='Normal Distribution')
plt.legend()
plt.show()
```


***

### 2 **Bernoulli Distribution**

- **Definition:** Models binary outcomes (success/failure, yes/no).
- **Formula:**

$$
P(X=1) = p, \quad P(X=0) = 1-p
$$

Where \$p\$ is the probability of success.
- **Business Impact:** Used in binary classification (e.g., churn prediction, email spam detection).
- **Python Example:**

```python
from scipy.stats import bernoulli
data = bernoulli.rvs(p=0.7, size=1000)
# 1 = success, 0 = failure
```


***

### 3 **Binomial Distribution**

- **Definition:** Models the number of successes in \$ n \$ independent Bernoulli trials.
- **Formula:**

$$
P(r) = \binom{n}{r} p^r (1-p)^{n-r}
$$

Where \$n\$ is number of trials, \$r\$ is number of successes, \$p\$ is probability of success.
- **Business Impact:** Used for modeling conversion rates, A/B testing, and event counts.
- **Python Example:**

```python
from scipy.stats import binom
binom_samples = binom.rvs(n=10, p=0.5, size=1000)
```


***

### 4 **Poisson Distribution**

- **Definition:** Models the number of events in a fixed interval, given a known average rate \$\lambda\$.
- **Formula:**

$$
P(r) = \frac{e^{-\lambda} \lambda^r}{r!}
$$
- **Business Impact:** Used for modeling rare events (e.g., website visits per minute, customer calls per hour, anomaly detection).
- **Python Example:**

```python
from scipy.stats import poisson
poisson_samples = poisson.rvs(mu=3, size=1000)
```


***

### 5 **Multinomial Distribution**

- **Definition:** Generalizes binomial to more than two categories; models counts for each category in \$n\$ trials.
- **Business Impact:** Used in multi-class classification, text classification, and NLP.
- **Python Example:**

```python
from numpy.random import multinomial
# 10 trials, probabilities for 3 categories
samples = multinomial(10, [0.2, 0.5, 0.3], size=1000)
```


***

### 6 **Beta Distribution**

- **Definition:** Models probabilities between 0 and 1; often used in Bayesian inference and A/B testing.
- **Formula:**

$$
f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
$$
- **Business Impact:** Used for modeling uncertainty in probabilities, e.g., conversion rates.
- **Python Example:**

```python
from scipy.stats import beta
x = np.linspace(0, 1, 100)
plt.plot(x, beta.pdf(x, 2, 5), label='Beta Distribution')
plt.legend()
plt.show()
```


***

### 7 **How to Choose the Right Distribution?**

- **Understand your data:** Is it continuous or discrete? Binary or multi-class?
- **Match the distribution to the business scenario:**
    - Use normal for continuous, symmetric data (e.g., heights, errors).
    - Use Bernoulli/binomial for binary or count data (e.g., conversions, clicks).
    - Use Poisson for event counts over time (e.g., calls per hour).
    - Use multinomial for multi-class outcomes (e.g., text categories).
    - Use beta for modeling probabilities and uncertainty.

***

## **Business Impact Summary Table**

| Distribution | Typical Use Case | Business Example |
| :-- | :-- | :-- |
| Normal | Continuous, symmetric data | Modeling sales residuals, test scores |
| Bernoulli | Binary outcomes | Churn prediction, fraud detection |
| Binomial | Count of successes in trials | Website conversions, A/B testing |
| Poisson | Event counts over time/space | Customer calls/hour, anomaly detection |
| Multinomial | Multi-class outcomes | Text classification, segmentation |
| Beta | Probabilities, uncertainty | Conversion rate modeling, Bayesian A/B |


***



## Normal (Gaussian) Distribution

1. **Definition and Formula**

The **normal distribution** (also called Gaussian) is a continuous probability distribution with a symmetric, bell-shaped curve. It is defined by its mean \$\mu\$ and standard deviation \$\sigma\$:

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- \$\mu\$: Center of the distribution (mean)
- \$\sigma\$: Spread (standard deviation)


2. **Key Properties**

- **Symmetry:** The curve is symmetric about the mean.
- **Empirical Rule:**
    - 68% of data within 1 standard deviation (\$\mu \pm 1\sigma\$)
    - 95% within 2 standard deviations (\$\mu \pm 2\sigma\$)
    - 99.7% within 3 standard deviations (\$\mu \pm 3\sigma\$)
- **Linear Transformations:** If \$X\$ is normal, so is \$aX + b\$.
- **Multivariate Normal:** Extends to multiple variables, modeling joint distributions with a mean vector and covariance matrix.


3. **Practical Examples**

- **Heights and Weights:** Human heights, weights, and many biological measurements are approximately normal.[^41_1][^41_2][^41_4]
- **Measurement Errors:** Errors in repeated measurements often follow a normal distribution.[^41_5][^41_1]
- **Test Scores:** Standardized test scores (e.g., IQ, SAT) are designed to be normally distributed.[^41_2][^41_4]


**Numerical Example**

Suppose adult male weights are normally distributed with \$\mu = 70\$ kg and \$\sigma = 5\$ kg. What proportion weigh more than 75 kg?

- **Step 1:** Calculate Z-score: \$Z = \frac{75 - 70}{5} = 1\$
- **Step 2:** Area to left of Z=1 is about 0.8413 (from Z-table)
- **Step 3:** Proportion above 75 kg: \$1 - 0.8413 = 0.1587\$ (about 15.87%)[^41_5]


4. **Applications in Machine Learning**

- **Statistical Inference:** Underpins hypothesis testing, confidence intervals, and parameter estimation.[^41_6][^41_5]
- **Regression:** Assumption of normally distributed errors in linear regression.
- **Clustering:** Gaussian Mixture Models (GMMs) use normal distributions to model clusters in data.[^41_6][^41_5]
- **Anomaly Detection:** Points far from the mean (in the tails) are flagged as anomalies.
- **Dimensionality Reduction:** Principal Component Analysis (PCA) assumes data is normally distributed for optimal performance.[^41_3][^41_6]
- **Kernel Methods:** Gaussian kernels in SVMs and Gaussian Processes define similarity between points.[^41_6]


5. **Python Example: Normal Distribution**

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate normal data
mu, sigma = 70, 5
samples = norm.rvs(loc=mu, scale=sigma, size=1000)

# Plot histogram and PDF
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Sampled Data')
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'k', linewidth=2, label='Normal PDF')
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
```


6. **Business Impact**

- **Quality Control:** Detect outliers in manufacturing.
- **Finance:** Model returns and risk.
- **A/B Testing:** Calculate p-values and confidence intervals.
- **Customer Segmentation:** GMMs for clustering customer behavior.


7. **Summary Table: Normal Distribution**

| Property | Value/Use Case |
| :-- | :-- |
| Shape | Symmetric, bell-shaped |
| Parameters | Mean (\$\mu$), Std Dev ($\sigma\$) |
| ML Applications | Regression, GMM, PCA, anomaly detection |
| Business Impact | Forecasting, quality control, risk |


***

### Real-World Examples of the Normal Distribution

The normal distribution is one of the most common and important probability distributions in statistics and data science. It models many real-world phenomena, especially those that result from the sum of many small, independent effects.

1. **Birthweight of Babies**

- **Example:** The birthweight of newborn babies in the U.S. is normally distributed with a mean of about 7.5 pounds.
- **Business Impact:** Hospitals and pediatricians use this distribution to identify outliers (low/high birthweight) and allocate resources for neonatal care.[^42_1]


2. **Height of Males**

- **Example:** The height of males in the U.S. is roughly normally distributed with a mean of 70 inches and a standard deviation of 3 inches.[^42_3][^42_6][^42_1]
- **Business Impact:** Clothing manufacturers use this to design sizes that fit most customers, and health professionals use it to assess growth patterns.


3. **IQ Scores**

- **Example:** IQ scores are designed to follow a normal distribution with a mean of 100 and a standard deviation of 15.[^42_9][^42_3]
- **Business Impact:** Educational psychologists use this to identify students who may need special support or advanced programs.


4. **Blood Pressure**

- **Example:** Blood pressure in the general population is approximately normally distributed, with a mean around 85 mmHg and a standard deviation of 20 mmHg.[^42_4][^42_7]
- **Business Impact:** Medical researchers use this to set thresholds for hypertension and to study population health.


5. **Test Scores (SAT, ACT, etc.)**

- **Example:** Standardized test scores like SAT and ACT are designed to be normally distributed, allowing for fair comparison across students.[^42_10][^42_1][^42_9]
- **Business Impact:** Universities use these distributions to set admission cutoffs and scholarships.


6. **Stock Market Returns**

- **Example:** Daily returns of stocks often approximate a normal distribution, especially for large, diversified portfolios.[^42_3]
- **Business Impact:** Financial analysts use this to estimate risk (volatility) and make investment decisions.


7. **Shoe Sizes**

- **Example:** Shoe sizes for males in the U.S. are roughly normally distributed with a mean of size 10 and a standard deviation of 1.[^42_1][^42_3]
- **Business Impact:** Retailers use this to manage inventory and predict demand for different sizes.

***

### Practical Calculation Example: Probability Using the Normal Distribution

**Scenario:** The average marks of candidates in an entrance test is 75, with a standard deviation of 7.5 marks. What is the probability of a randomly selected candidate having marks less than 90?

- **Step 1:** Calculate the z-score:

$$
z = \frac{90 - 75}{7.5} = 2.00
$$
- **Step 2:** Look up the probability for \$z = 2.00\$ in the standard normal table: \$P(Z < 2.00) = 0.9772\$.
- **Interpretation:** There is a 97.7% chance that a randomly selected candidate will have marks less than 90.[^42_2]

***

**Python Example: Calculating Probabilities**

```python
from scipy.stats import norm
# Mean and standard deviation
mu = 75
sigma = 7.5
# Probability of marks less than 90
prob = norm.cdf(90, loc=mu, scale=sigma)
print(f"Probability of marks < 90: {prob:.4f}")  # Output: 0.9772
```


***

### Key Takeaways

- The normal distribution is used to model many natural and social phenomena: heights, weights, test scores, blood pressure, and more.
- It allows us to calculate probabilities, identify outliers, and make informed business and healthcare decisions.
- In practice, you can use the normal distribution to estimate the percentage of a population within a certain range, set thresholds, and guide resource allocation.

## Bernoulli Distribution

## 1. **Definition and Formula**

The **Bernoulli distribution** models a single trial with only two possible outcomes: success (1) or failure (0). It is parameterized by \$p\$, the probability of success.

- **Probability Mass Function:**

$$
P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

Where \$p\$ is the probability of success, \$1-p\$ is the probability of failure.


## 2. **Key Properties**

- **Binary outcomes:** Only two possible results (e.g., heads/tails, pass/fail, yes/no).
- **Mean:** \$\mu = p\$
- **Variance:** \$\sigma^2 = p(1-p)\$
- **Symmetry:** If \$p = 0.5\$, outcomes are equally likely; otherwise, the distribution is skewed.


## 3. **Practical Examples**

- **Coin Toss:** Probability of heads (\$p = 0.5\$).
- **Quality Control:** Whether a product passes (1) or fails (0) inspection.
- **Medical Testing:** Whether a patient tests positive (1) or negative (0) for a disease.
- **Customer Churn:** Whether a customer leaves (1) or stays (0).
- **Email Spam Detection:** Whether an email is spam (1) or not (0).


## 4. **Applications in Machine Learning**

- **Binary Classification:** Logistic regression, Bernoulli Naive Bayes, and other algorithms model the probability of a binary outcome using the Bernoulli distribution.[^43_1][^43_6]
- **A/B Testing:** Each user either converts (1) or does not (0); conversion rate is modeled as Bernoulli.
- **Hypothesis Testing:** Used to test proportions (e.g., success rate in a sample).
- **Simulation:** Monte Carlo simulations of binary events.


## 5. **Python Example: Bernoulli Distribution**

```python
from scipy.stats import bernoulli
import numpy as np
import matplotlib.pyplot as plt

# Simulate 1000 Bernoulli trials with p=0.3
p = 0.3
data = bernoulli.rvs(p, size=1000)

# Plot the results
plt.hist(data, bins=2, rwidth=0.8)
plt.xticks([0, 1])
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.title('Bernoulli Distribution (p=0.3)')
plt.show()

# Calculate mean and variance
mean = np.mean(data)
var = np.var(data)
print(f"Sample mean: {mean:.2f}, Sample variance: {var:.2f}")
```


## 6. **Business Impact**

- **Risk Analysis:** Model probability of success/failure in investments, product launches, or process improvements.
- **Quality Control:** Estimate defect rates and optimize manufacturing.
- **Marketing:** Predict probability of customer conversion or churn.
- **Healthcare:** Assess treatment effectiveness (e.g., cured/not cured).


## 7. **Summary Table: Bernoulli Distribution**

| Property | Value/Use Case |
| :-- | :-- |
| Shape | Two spikes at 0 and 1 |
| Parameter | Probability of success (\$ p \$) |
| ML Applications | Binary classification, A/B testing |
| Business Impact | Risk, conversion, quality control |


***

Would you like to dive deeper into the next distribution (Binomial), or see more real-world examples and code for Bernoulli distribution applications?
<span style="display:none">[^43_2][^43_3][^43_4][^43_5][^43_7]</span>

<div align="center">⁂</div>

[^43_1]: https://www.datacamp.com/tutorial/bernoulli-distribution

[^43_2]: https://www.cs.ubc.ca/~schmidtm/Courses/440-W22/L2.pdf

[^43_3]: https://www.rajivgopinath.com/blogs/statistics-and-data-science-hub/bernoulli-distribution-explained

[^43_4]: https://www.simplilearn.com/tutorials/data-science-tutorial/bernoulli-distribution

[^43_5]: https://www.machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/

[^43_6]: https://careerfoundry.com/en/blog/data-analytics/what-is-bernoulli-distribution/

[^43_7]: https://towardsdatascience.com/bernoulli-naive-bayes-explained-a-visual-guide-with-code-examples-for-beginners-aec39771ddd6/


---

## Real-World Examples and Code for Bernoulli Distribution Applications

The Bernoulli distribution is fundamental for modeling binary (yes/no, success/failure) outcomes in data science, business, and research. Here are practical examples and code to illustrate its use:

***

1. **Coin Tossing**

- **Scenario:** Flipping a fair coin (success = heads, failure = tails).
- **Business/Science Use:** Simulating random events, teaching probability basics.
- **Python Example:**

```python
from scipy.stats import bernoulli
import numpy as np
# Simulate 1000 coin tosses (p=0.5)
coin_flips = bernoulli.rvs(p=0.5, size=1000)
print(f'Heads proportion: {np.mean(coin_flips):.2f}')
```


***

2. **Medical Research (Drug Trials)**

- **Scenario:** Each patient either responds to a new drug (1) or does not (0).
- **Business/Science Use:** Estimating treatment effectiveness, calculating cure rates, and designing clinical trials.[^44_1][^44_2][^44_6]
- **Python Example:**

```python
# Simulate 1000 patients, 30% success rate
drug_response = bernoulli.rvs(p=0.3, size=1000)
print(f'Success rate: {np.mean(drug_response):.2f}')
```


***

3. **Market Analysis (Customer Purchase)**

- **Scenario:** A customer either buys a product (1) or not (0).
- **Business Use:** Modeling conversion rates, predicting sales, and optimizing marketing strategies.[^44_2][^44_6][^44_1]
- **Python Example:**

```python
# Simulate 1000 website visitors, 8% conversion rate
purchases = bernoulli.rvs(p=0.08, size=1000)
print(f'Conversion rate: {np.mean(purchases):.2f}')
```


***

4. **Quality Control (Pass/Fail Inspection)**

- **Scenario:** Each product passes (1) or fails (0) inspection.
- **Business Use:** Estimating defect rates, improving manufacturing processes, and setting quality thresholds.[^44_6][^44_1]
- **Python Example:**

```python
# Simulate 1000 products, 98% pass rate
passes = bernoulli.rvs(p=0.98, size=1000)
print(f'Pass rate: {np.mean(passes):.2f}')
```


***

5. **Binary Classification in Machine Learning**

- **Scenario:** Predicting if an email is spam (1) or not (0), or if a transaction is fraudulent (1) or legitimate (0).
- **ML Use:** Logistic regression, Bernoulli Naive Bayes, and other binary classifiers assume the target variable follows a Bernoulli distribution.[^44_2][^44_6]
- **Python Example:**

```python
# Simulate binary labels for spam detection (20% spam)
spam_labels = bernoulli.rvs(p=0.2, size=1000)
print(f'Spam rate: {np.mean(spam_labels):.2f}')
```


***

6. **A/B Testing and Hypothesis Testing**

- **Scenario:** Each user either clicks an ad (1) or not (0) in an A/B test.
- **Business Use:** Comparing conversion rates between two marketing strategies, determining statistical significance.[^44_1][^44_6][^44_2]
- **Python Example:**

```python
# Simulate 1000 users, 12% click rate
clicks = bernoulli.rvs(p=0.12, size=1000)
print(f'Click-through rate: {np.mean(clicks):.2f}')
```


***

7. **Simulation and Risk Analysis**
- **Scenario:** Modeling the probability of a system failure (1) or success (0) in risk analysis or Monte Carlo simulations.[^44_6]
- **Business Use:** Estimating risk, planning for rare events, and stress-testing systems.

***

**Key Takeaways**

- The Bernoulli distribution is the foundation for modeling any process with two possible outcomes.
- It is widely used in business (conversion, churn), healthcare (treatment success), manufacturing (defect detection), and machine learning (binary classification).
- In Python, you can simulate and analyze Bernoulli processes easily with `scipy.stats.bernoulli`.

## Binomial Distribution

1. **Definition and Formula**

The **binomial distribution** models the number of successes in a fixed number of independent Bernoulli trials (each with two possible outcomes: success or failure) and a constant probability of success.

- **Probability Mass Function:**

$$
P(X = x) = \binom{n}{x} p^x (1-p)^{n-x}
$$

Where:
    - \$n\$: number of trials
    - \$x\$: number of successes (can be 0, 1, ..., n)
    - \$p\$: probability of success in a single trial
    - \$1-p\$: probability of failure
    - \$\binom{n}{x} = \frac{n!}{x!(n-x)!}\$: number of ways to choose \$x\$ successes from \$n\$ trials[^45_2][^45_3][^45_5][^45_7]


2. **Key Properties**

- **Mean:** \$\mu = np\$
- **Variance:** \$\sigma^2 = np(1-p)\$
- **Standard Deviation:** \$\sigma = \sqrt{np(1-p)}\$
- **Discrete:** Only integer values from 0 to \$n\$
- **Symmetry:** If \$p = 0.5\$, distribution is symmetric; otherwise, it is skewed.[^45_3][^45_9]


3. **Practical Examples**

- **Coin Tosses:** Probability of getting exactly 6 heads in 20 flips.
- **Quality Control:** Number of defective items in a batch of 100, given a defect rate.
- **Marketing:** Number of customers who make a purchase out of 50 contacted, with a known conversion rate.
- **Finance:** Number of loan defaults in a portfolio of 200 loans, given a default probability.


4. **Step-by-Step Example**

**Scenario:** A coin is tossed 5 times. What is the probability of getting exactly 2 heads?

- $n = 5$, $p = 0.5$, $x = 2$

$$
P(X = 2) = \binom{5}{2} (0.5)^2 (0.5)^{5-2} = \frac{5!}{2!3!} \times 0.25 \times 0.125 = 10 \times 0.03125 = 0.3125
$$

- **Interpretation:** There is a 31.25% chance of getting exactly 2 heads in 5 tosses.[^45_5][^45_2]


5. **Python Example: Binomial Distribution**

```python
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

# Probability of exactly 2 heads in 5 tosses
n, p, x = 5, 0.5, 2
prob = binom.pmf(x, n, p)
print(f"P(X=2): {prob:.4f}")  # Output: 0.3125

# Probability of at least 4 heads
prob_at_least_4 = binom.pmf(4, n, p) + binom.pmf(5, n, p)
print(f"P(X>=4): {prob_at_least_4:.4f}")

# Plotting the distribution for n=20, p=0.5
n, p = 20, 0.5
x = np.arange(0, n+1)
probs = binom.pmf(x, n, p)
plt.bar(x, probs)
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title('Binomial Distribution (n=20, p=0.5)')
plt.show()
```


6. **Business Impact**

- **Risk Management:** Estimate probability of a certain number of defaults in a loan portfolio.
- **A/B Testing:** Calculate probability of observed conversions under null hypothesis.
- **Quality Assurance:** Predict number of defective products in a batch.
- **Resource Planning:** Estimate number of positive responses in a marketing campaign.


7. **Summary Table: Binomial Distribution**

| Property | Value/Use Case |
| :-- | :-- |
| Shape | Discrete, 0 to n successes |
| Parameters | n (trials), p (success probability) |
| ML Applications | Event counts, hypothesis testing |
| Business Impact | Risk, conversion, quality control |


***

## Poisson Distribution

1. **Definition and Formula**

The **Poisson distribution** is a discrete probability distribution that models the number of times an event occurs in a fixed interval of time or space, given a known constant mean rate and independent occurrences.[^46_1][^46_2][^46_6]

**Probability Mass Function:**

$$
P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}
$$

Where:

    - \$k\$: number of occurrences (0, 1, 2, ...)
    - \$\lambda\$: average rate (mean number of events per interval)
    - \$e\$: Euler’s number (≈ 2.718)


 2. **Key Properties**

- **Mean and Variance:** Both equal to \$\lambda\$
- **Discrete:** Only non-negative integer values
- **Right-skewed** for small \$\lambda\$; becomes more symmetric as \$\lambda\$ increases
- **Events are independent** and occur at a constant average rate[^46_6][^46_1]


3. **Practical Examples**

- **Call Center:** Number of calls received per hour
- **Website Analytics:** Number of user logins per minute
- **Manufacturing:** Number of defects per meter of fabric
- **Biology:** Number of mutations in a DNA strand per unit length
- **Finance:** Number of trades per second on a stock exchange


4. **Step-by-Step Example**

**Scenario:** A call center receives an average of 4 calls per minute (\$\lambda = 4\$). What is the probability of receiving exactly 6 calls in a minute?

$$
P(X = 6) = \frac{e^{-4} \cdot 4^6}{6!} = \frac{0.0183 \cdot 4096}{720} \approx 0.1042
$$

- **Interpretation:** There is a 10.4% chance of receiving exactly 6 calls in a minute.


5. **Python Example: Poisson Distribution**

```python
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

# Probability of exactly 6 calls in a minute (lambda=4)
prob = poisson.pmf(6, mu=4)
print(f"P(X=6): {prob:.4f}")  # Output: 0.1042

# Probability of 0, 1, ..., 10 calls
x = np.arange(0, 11)
probs = poisson.pmf(x, mu=4)
plt.bar(x, probs)
plt.xlabel('Number of Calls')
plt.ylabel('Probability')
plt.title('Poisson Distribution (lambda=4)')
plt.show()
```


6. **Business Impact**

- **Resource Planning:** Predict peak loads (e.g., staffing for call centers, server capacity for websites)
- **Quality Control:** Estimate number of defects in production
- **Risk Management:** Model rare events (e.g., insurance claims, system failures)
- **Healthcare:** Model number of arrivals at an emergency room per hour


7. **Summary Table: Poisson Distribution**

| Property | Value/Use Case |
| :-- | :-- |
| Shape | Discrete, right-skewed (small $$
\lambda
$$) |
| Parameter | $$
\lambda
$$ (mean rate) |
| ML Applications | Event counts, anomaly detection |
| Business Impact | Staffing, quality, risk, forecasting |


