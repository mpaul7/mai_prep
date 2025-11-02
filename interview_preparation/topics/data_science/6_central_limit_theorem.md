# Central Limit Theorem (CLT) \& Law of Large Numbers

## 6.1 **Central Limit Theorem (CLT)**

- **Definition:** As the sample size increases, the distribution of sample means approaches a normal (bell-shaped) distribution, regardless of the population's actual distribution.
- **Business Impact:** The CLT justifies the use of normal-based confidence intervals and hypothesis tests for sample means—even if the population is skewed or not normal. This drives quality control, market research, financial forecasting, and much more in business.


**Real-World CLT Applications**

- **Manufacturing:** Companies estimate the average product quality (e.g., lifespan of light bulbs) using sample averages, allowing early detection of quality issues and maintaining product standards.
- **Textile:** Workers measure fabric thickness at many points; the average thickness tends to be normally distributed due to CLT, enabling easy quality assessment.
- **Food Processing:** Random can weights from batches are averaged; the CLT estimates overall quality and highlights process issues.
- **Finance:** Analysts sample stock returns, and means of those samples become normally distributed, supporting risk models, forecasts, and portfolio management.


**Practical CLT Example (Code)**

Let's simulate means of samples from a skewed (exponential) distribution and observe the normality of their means:

```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(11)

# Generate a right-skewed population (exponential)
pop = np.random.exponential(scale=2, size=10000)
plt.hist(pop, bins=40, alpha=0.6, label='Population (skewed)')

# Take 1000 samples of size 50, and store their means
sample_means = [np.mean(np.random.choice(pop, size=50, replace=False)) for _ in range(1000)]
plt.hist(sample_means, bins=40, alpha=0.6, label='Sample Means (CLT)')

plt.legend()
plt.title('CLT: Population vs. Sample Means')
plt.show()
```

- **Interpretation:** Although the original data is skewed, the histogram of sample means is bell-shaped—showing the CLT in action!

***

6.2 **Law of Large Numbers (LLN)**

- **Definition:** As you increase the number of independent observations, the sample mean gets closer and closer to the population mean.
- **Business Impact:** This principle underlies averaging premiums in insurance, estimating average customer spending, and sports statistics.


**Real-World LLN Applications**

- **Coin Tosses:** The average of many thousand coin tosses closely matches the theoretical mean (0.5 for fairness).
- **Sports Stats:** Seasonal player stats (like completion percentage) stabilize and approach the true ability as games accumulate.
- **Insurance:** Pooling risk over thousands of policyholders lets insurers accurately predict average costs and set fair rates.


**Practical LLN Example (Code)**

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# Example: Simulated dice rolls (theoretical mean = 3.5)
avgs = []
for i in range(1, 5001):
    rolls = np.random.randint(1, 7, size=i) # Roll i dice
    avgs.append(np.mean(rolls))
plt.plot(avgs)
plt.axhline(3.5, color='red', linestyle='--', label='True Mean')
plt.xlabel('Number of Rolls')
plt.ylabel('Sample Mean')
plt.title('Law of Large Numbers: Sample Mean Converges')
plt.legend()
plt.show()
```

- **Interpretation:** As more dice are rolled, the sample mean curve approaches the true mean of 3.5—demonstrating the LLN.

***

6.3 **Summary Table: CLT \& LLN Business Impact**

| Topic | What It Enables | Example |
| :-- | :-- | :-- |
| CLT | Use normal stats for sample means | Quality control averages, market forecasting |
| LLN | Reliable averages with big samples | Risk insurance, sports stats, customer analytics |


***