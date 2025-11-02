# Statistics Fundamentals for Data Science & Data Analytics


## Inferential Statistics

Inferential statistics allow you to make predictions, generalizations, and decisions about a population based on sample data. This is essential in data science, where working with entire populations is often impractical.

Inferential statistics is a branch of statistics focused on drawing conclusions, making predictions, or generalizing about a broader population based on data from a sample. Instead of just summarizing what has been observed (like descriptive statistics), inferential statistics lets you make evidence-based guesses about unobserved values, relationships, or the effects of interventions.

**How It Differs from Descriptive Statistics:**
- **Descriptive statistics** (mean, median, standard deviation, etc.) summarize and describe the key features of a data set—what you actually observed. They provide a snapshot of the sample and do **not** allow you to make generalizations beyond those data.
- **Inferential statistics** goes further by using a sample to infer insights about the entire population. It employs probability theory, estimation, and hypothesis testing to answer questions like: 
  - "Is there evidence that this treatment works in the population?"
  - "What is the likely range for the true population mean?"
  - "Are two groups significantly different?"

**Why Inferential Statistics Matter?**

- **Generalization:** Make predictions about unseen data.
- **Decision Making:** Test business hypotheses, validate models.
- **Uncertainty Quantification:** Express results with confidence, not just point estimates.

**Examples of Inferential Statistics:** Hypothesis testing, confidence intervals, regression analysis, and correlation analysis.

**Summary Table: Descriptive vs Inferential Statistics**

| Aspect | Descriptive Statistics | Inferential Statistics |
| :-- | :-- | :-- |
| Purpose | Summarize sample data | Generalize to population |
| Techniques | Mean, median, mode, SD, IQR | Hypothesis testing, CI, regression |
| Data Used | Known sample | Sample to infer about population |
| Output | Facts, charts, tables | Probabilities, predictions, estimates |
| Uncertainty | None | Always present (sampling error, CI) |


**Key point:**  
Descriptive statistics describe your data; inferential statistics help you use your data to learn about the world beyond your sample.

### Core Techniques of Inferential Statistics

The main tools and techniques used in inferential statistics allow you to make generalizations or decisions about a population based on a sample. The core techniques include:

1. **Hypothesis Testing:**  
   Used to determine whether there is enough evidence in a sample to support a specific claim about a population. Common tests include the t-test, z-test, chi-squared test, and ANOVA.

2. **Confidence Intervals:**  
   Provide a range of values within which the true population parameter (like mean or proportion) is expected to lie with a certain degree of confidence (e.g., 95%).

3. **Regression Analysis:**  
   Helps model the relationship between one or more independent variables and a dependent variable. Widely used for prediction and assessing variable impact (e.g., linear and logistic regression).

4. **Correlation Analysis:**  
   Measures the strength and direction of association between two quantitative variables (e.g., Pearson’s or Spearman’s correlation).

5. **Analysis of Variance (ANOVA):**  
   A technique to compare means across multiple groups to determine if at least one group mean is different from the others.

6. **Estimation:**  
   Inference about the value of population parameters through point estimates and interval estimates.

7. **Sampling Methods & Resampling Techniques:**  
   Methods like bootstrapping and permutation tests are often used to make inferences when theoretical assumptions are hard to meet.

**Summary Table:**

| Technique                | Purpose                                 | Example Use Case                 |
|--------------------------|-----------------------------------------|----------------------------------|
| Hypothesis Testing       | Test claims about a population          | Is the new drug effective?       |
| Confidence Intervals     | Estimate range for population parameter | What is the true mean salary?    |
| Regression Analysis      | Model relationships, predict outcomes   | Predict sales from ad spend      |
| Correlation Analysis     | Assess association between variables    | Are height and weight related?   |
| ANOVA                    | Compare means across groups             | Do different diets affect weight?|

These methods are the foundation of inferential statistics and enable data scientists to draw conclusions, quantify uncertainty, and guide decision-making based on data.


***


## Core Techniques Details

### 1. Hypothesis Testing
**Hypothesis Testing: An Overview**

**1. What is Hypothesis Testing?**

Hypothesis testing is a fundamental statistical technique used to make inferences about populations based on sample data. It answers questions like: "Does this treatment have an effect?" or "Are these two groups statistically different?"

Hypothesis testing is a structured method for evaluating assumptions (hypotheses) about a population using data from a sample. It helps determine whether observed effects are real or simply due to random chance.

**2. Core Concepts & Steps**

- **Null Hypothesis ($H_0$):**  
  A skeptical assumption—usually states there is *no effect*, *no difference*, or *no association*.  
  _Example:_ The mean income after training is the same as before, $H_0: \mu_\text{after} = \mu_\text{before}$

- **Alternative Hypothesis ($H_1$ or $H_a$):**  
  What you want to prove—posits a real effect or difference exists.  
  _Example:_ The mean income after training is different, $H_1: \mu_\text{after} \ne \mu_\text{before}$

- **Test Statistic:**  
  A value calculated from your sample that summarizes how much the data deviate from what would be expected under $H_0$. Common statistics: $z$, $t$, $\chi^2$, $F$.

- **P-value:**  
  The probability of observing data as extreme as your sample, assuming $H_0$ is true. Low p-values indicate evidence against $H_0$.

- **Significance Level ($\alpha$):**  
  The threshold to decide if a p-value is "small enough" (e.g., $\alpha = 0.05$).

**General Steps:**
1. **State hypotheses:** Null ($H_0$) and alternative ($H_1$).
2. **Set $\alpha$:** Commonly 0.05 or 0.01.
3. **Choose a test & compute test statistic:** (e.g., $t$-test, $z$-test, ANOVA)
4. **Find the p-value.**
5. **Decision:**  
   - If $p$ < $\alpha$, **reject $H_0$** (evidence for $H_1$).
   - If $p$ ≥ $\alpha$, **fail to reject $H_0$** (insufficient evidence).

**3. Common Hypothesis Test Formulas**

- **For one-sample $z$-test for the mean:**

  $$
  z = \frac{\overline{x} - \mu_0}{\sigma/\sqrt{n}}
  $$

  - $\overline{x}$: sample mean  
  - $\mu_0$: hypothesized population mean  
  - $\sigma$: population standard deviation  
  - $n$: sample size

- **For one-sample $t$-test (unknown $\sigma$):**

  $$
  t = \frac{\overline{x} - \mu_0}{s/\sqrt{n}}
  $$
  
  - $s$: sample standard deviation

**4. Interpreting Results**

- **Reject $H_0$:** There is enough statistical evidence to support the alternative hypothesis.
- **Fail to reject $H_0$:** There is not enough evidence to support the alternative; the result could be due to random variation.

*Remember: "Failing to reject" does not prove $H_0$ is true, only that we don't have enough evidence against it.*

**5. Use Cases for Hypothesis Testing**

- Comparing group means (A/B testing, clinical trials, marketing experiments)
- Determining whether a relationship exists between variables (correlation, regression)
- Evaluating whether distributions are different (e.g., non-parametric tests)
- Quality control (is defect rate within acceptable limits?)

**Summary Table of Common Tests:**

| Test Type               | Used For                             | Example                                    |
|-------------------------|--------------------------------------|--------------------------------------------|
| One-sample $t$-test     | Mean vs. known value                 | Is average height different from 170cm?     |
| Two-sample $t$-test     | Compare means of two groups          | Do test and control group differ?           |
| Paired $t$-test         | Compare means of paired samples      | Before and after training                   |
| Chi-squared test        | Compare categorical distributions    | Are gender and buying habits related?       |
| ANOVA                   | Compare means across >2 groups       | Impact of 3 diets on weight                 |
| Proportion $z$-test     | Compare population proportions       | Did click-through rate increase?            |

**Visual Flow**

1. **Question** → 2. **Hypotheses** → 3. **Select Test** → 4. **Compute Statistic** → 5. **P-value** → 6. **Decision**

---

**In practice, hypothesis testing provides a framework for making data-driven decisions under uncertainty, quantifying risk, and supporting or refuting claims with evidence.**

**Relating Hypothesis Testing to Real-World Examples**

1. **Real-World Applications of Hypothesis Testing**

Hypothesis testing is a powerful tool for making data-driven decisions in many fields. Here are some concrete examples:

**Business \& Marketing**

- **Advertising Effectiveness:** A company wants to know if increasing digital ad spend leads to higher sales. They compare sales before and after the campaign:
    - **Null hypothesis ($H_0$):** Mean sales before = mean sales after
    - **Alternative hypothesis ($H_1$):** Mean sales after > mean sales before
    - If the p-value is less than 0.05, they reject $H_0$ and conclude the campaign increased sales.[^17_1][^17_3]


**Manufacturing \& Quality Control**

- **Process Improvement:** A factory tests if a new method reduces defective products. They compare defect rates before and after:
    - **$H_0$:** Defect rate before = defect rate after
    - **$H_1$:** Defect rate after ≠ defect rate before
    - A low p-value means the new method significantly changed the defect rate.[^17_4][^17_6]


**Medicine \& Healthcare**

- **Drug Effectiveness:** Researchers test if a new drug improves patient outcomes compared to a placebo:
    - **$H_0$:** No difference in outcomes
    - **$H_1$:** Drug leads to better outcomes
    - A small p-value supports the drug's effectiveness.[^17_2][^17_3]


**Education \& Social Science**

- **Teaching Methods:** Schools test if project-based learning improves test scores compared to traditional methods.[^17_3]


**Environmental Science**

- **Pollution Impact:** Scientists test if a new water management strategy improves river health by comparing data before and after implementation.[^17_5]


**Significance of the p-Value**

- The **p-value** quantifies the probability of observing your data (or something more extreme) if the null hypothesis is true.
- **Low p-value (typically < 0.05):** Strong evidence against $H_0$; you reject the null hypothesis.
- **High p-value:** Not enough evidence to reject $H_0$; results could be due to chance.
- **Interpretation:**
    - If p-value = 0.03, there's a 3% chance you'd see these results if $H_0$ were true. This is considered statistically significant.

**Why It Matters in Decision-Making**

- Hypothesis testing helps avoid making decisions based on random chance or anecdotal evidence.
- The p-value provides a standardized way to judge whether an observed effect is likely to be real or just noise.
- In business, medicine, and science, this leads to more reliable, reproducible decisions and policies.

***

**Quick Check:**

- Can you think of a situation in your field where hypothesis testing could help validate a new method or product?
- How would you set up the null and alternative hypotheses?



**Step-by-Step Example: Hypothesis Testing in Context**

Let's walk through a real-world hypothesis test, interpret the p-value, and see how each step connects to practical decision-making.

***

 Example Scenario: Comparing Two Teaching Methods

Suppose a school wants to know if a new teaching method improves student test scores compared to the traditional method.

**Step 1: State Hypotheses**

- **Null hypothesis ($H_0$):** The mean score with the new method is equal to the mean score with the traditional method ($\mu_{new} = \mu_{trad}$).
- **Alternative hypothesis ($H_1$):** The mean score with the new method is higher ($\mu_{new} > \mu_{trad}$).


**Step 2: Collect Data**

- Randomly assign students to two groups: one uses the new method, one uses the traditional method.
- After a semester, collect test scores from both groups.


**Step 3: Choose the Test \& Significance Level**

- Use a two-sample t-test (since we're comparing means of two independent groups).
- Set significance level $\alpha = 0.05$ (5%).


**Step 4: Calculate Test Statistic \& p-value**

- Compute the t-statistic using the formula:

$$
t = \frac{\overline{x}_{new} - \overline{x}_{trad}}{\sqrt{\frac{s_{new}^2}{n_{new}} + \frac{s_{trad}^2}{n_{trad}}}}
$$
- Use statistical software (e.g., Python's `scipy.stats.ttest_ind`) to get the t-statistic and p-value.


**Step 5: Make a Decision**

- **Interpret the p-value:**
    - If p-value $\leq \alpha$ (e.g., p = 0.03), reject $H_0$. There is statistically significant evidence that the new method improves scores.
    - If p-value $> \alpha$ (e.g., p = 0.12), fail to reject $H_0$. The observed difference could be due to chance.


**Step 6: Report \& Interpret Results**

- "With a p-value of 0.03, we reject the null hypothesis and conclude that the new teaching method leads to higher average test scores than the traditional method."
- **Practical significance:** Even if statistically significant, consider if the improvement is large enough to matter in practice.

***

What Does the p-value Mean?

- The p-value is the probability of observing a result as extreme as (or more extreme than) your sample, assuming the null hypothesis is true.
- **Low p-value (< 0.05):** Strong evidence against $H_0$; the effect is unlikely due to random chance.
- **High p-value:** Not enough evidence to claim a real effect; the difference could be random.

***

**Quick Review**

- **Hypothesis testing** helps you make objective decisions using data.
- **p-value** quantifies the strength of evidence against the null hypothesis.
- Always consider both statistical and practical significance when interpreting results.

Would you like to see the Python code for this example, or try interpreting p-values in other scenarios? Let me know how you'd like to practice next!


---

**Practical Example: Interpreting the p-Value in a Sales and Advertisement Scenario**

Let's walk through a real-world example of using hypothesis testing and interpreting the p-value in the context of sales with and without advertisement.

***


**Scenario: Does Advertising Increase Sales?**

Suppose a company wants to know if running an advertising campaign actually increases their sales. They collect sales data for two periods:

- **Period 1:** No advertisement (control group)
- **Period 2:** With advertisement (treatment group)

They want to test if the mean sales during the advertisement period are significantly higher than during the period without ads.

***

Step-by-Step Hypothesis Test

1. **State the Hypotheses**

- **Null hypothesis ($H_0$):** Advertising does not increase sales (mean sales with ads = mean sales without ads).
- **Alternative hypothesis ($H_1$):** Advertising increases sales (mean sales with ads > mean sales without ads).


2. **Collect Data**

- Gather sales figures for both periods (e.g., daily or weekly sales).


3. **Choose the Test \& Significance Level**

- Use a two-sample t-test (comparing means of two independent groups).
- Set significance level $\alpha = 0.05$ (5%).


4. **Calculate the Test Statistic and p-Value**

- Run the t-test using your data. Suppose you get a **p-value of 0.03**.


5. **Interpret the p-Value**

- The **p-value** is the probability of observing a difference in sales as large as (or larger than) what you found, **if the null hypothesis were true** (i.e., if advertising had no effect).[^20_2][^20_3][^20_5]
- In this case, **p = 0.03** means there is a 3% chance you'd see this much of a sales increase just by random chance, if advertising actually had no effect.


6. **Decision**

- Since **0.03 < 0.05**, you **reject the null hypothesis**.
- **Conclusion:** There is statistically significant evidence that advertising increases sales.[^20_3][^20_5]

***

Key Takeaways

- **Low p-value (≤ 0.05):** Strong evidence against the null hypothesis; the observed sales increase is unlikely due to chance.
- **High p-value (> 0.05):** Not enough evidence to claim advertising had an effect; the difference could be random.

> **In practice:** If your p-value is 0.03, you can confidently report to stakeholders that the advertising campaign had a statistically significant positive impact on sales, with less than a 5% probability that this result is due to random variation alone.[^20_5][^20_3]



Would you like to see the Python code for running this t-test, or try interpreting p-values in other business scenarios?


### 1a. Practical Hypothesis Testing Examples

**1. Population Parameter**

- **Definition:** A method to test assumptions (hypotheses) about a population parameter.
- **Process:**

1. State null ($H_0$) and alternative ($H_1$) hypotheses.
2. Choose significance level ($\alpha$, often 0.05).
3. Calculate test statistic (e.g., $z$, $t$).
4. Find p-value and compare to $\alpha$.
5. Accept or reject $H_0$.
- **Example:** Testing if a new drug is more effective than the current standard.

**2. Real-Life Example of Hypothesis Testing:**

*Suppose a tech company wants to determine if a new website design increases the average time users spend on the site compared to the old design.*

- **Step 1: Define Hypotheses**
    - Null hypothesis ($H_0$): The mean time on site with the new design is **equal** to the mean with the old design.
    - Alternative hypothesis ($H_1$): The mean time on site with the new design is **greater** than with the old design.

- **Step 2: Collect Data**
    - Randomly assign a sample of users to see the old design and a separate random sample to see the new design.
    - Record the amount of time (in minutes) each user spends on the site.

- **Step 3: Choose a Test & Analyze**
    - Use a two-sample $t$-test to compare the means of the two groups.

- **Step 4: Decision**
    - Calculate the $t$-statistic and associated p-value.
    - If the p-value is less than the chosen significance level (e.g., $\alpha=0.05$), **reject** the null hypothesis.

- **Step 5: Interpret Results**
    - If you reject $H_0$, you have statistical evidence that the new design increases user engagement.
    - If you fail to reject $H_0$, there isn’t enough evidence to conclude the new design is better.

**Summary Table:**

|            | Old Design | New Design |
|------------|------------|------------|
| Sample Size|     200    |     200    |
| Mean Time (min) | 4.2  |    5.0     |
| SD (min)   |   1.5      |   1.7      |

*If the two-sample $t$-test yields a p-value of 0.01, you would conclude there is a statistically significant increase in user time with the new design.*

**3. Alternative Example: Hypothesis Testing in a Sales Scenario**

*Imagine a retail company wants to evaluate whether an advertising campaign increases the average weekly sales of a particular product.*

- **Step 1: Define Hypotheses**
    - Null hypothesis ($H_0$): The mean weekly sales after the advertisement are **equal to** the mean weekly sales before the advertisement.
    - Alternative hypothesis ($H_1$): The mean weekly sales after the advertisement are **greater than** before the advertisement.

- **Step 2: Collect Data**
    - Record weekly sales for a certain period **before** launching the advertisement (e.g., 10 weeks).
    - Record weekly sales for the same period **after** launching the advertisement (next 10 weeks).

- **Step 3: Choose Test & Analyze**
    - Use a **paired $t$-test** (if the sales are recorded for the same stores before and after) or a **two-sample $t$-test** (if comparing separate samples of stores/weeks).

- **Step 4: Decision**
    - Calculate the $t$-statistic and corresponding p-value.
    - If the p-value is less than the chosen significance level (e.g., $\alpha = 0.05$), **reject** the null hypothesis.

- **Step 5: Interpret Results**
    - **Reject $H_0$**: There is evidence that the advertisement increased the average weekly sales.
    - **Fail to reject $H_0$**: No evidence that the advertisement had a significant effect.

**Summary Table:**

|               | Before Ad | After Ad  |
|---------------|-----------|-----------|
| Weeks         |   10      |    10     |
| Mean Sales    | $500      |  $575     |
| Standard Dev. | $50       |  $60      |

*If the paired $t$-test produces a p-value of 0.03, you would conclude the advertisement significantly increased average weekly product sales.*


---

This is a practical example of using hypothesis testing to support a business decision using data.

---

### 1b. Python Example: Hypothesis Testing

```python
import numpy as np
from scipy import stats

# Example: Test if sample mean differs from population mean
data = [4, 8, 9, 10, 6, 12, 14, 4, 5, 3, 4]
population_mean = 7

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(data, population_mean)
print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")

# Confidence interval for mean
mean = np.mean(data)
std_err = stats.sem(data)
conf_int = stats.t.interval(0.95, len(data)-1, loc=mean, scale=std_err)
print(f"95% Confidence Interval: {conf_int}")
```

### 2. A/B Testing

- **Definition:**  
  A/B testing (also called split testing) is a randomized experimentation process in which two versions (A and B) of a single variable (web page, email, feature, etc.) are compared to determine which performs better on a given outcome (conversion rate, click-through, etc.).

- **How it's implemented:**  
  1. **Define Metric:** Decide what to measure (e.g., conversion rate).
  2. **Split Group:** Randomly divide users into two groups:
      - **Group A:** Control group (shown current version)
      - **Group B:** Treatment group (shown new version)
  3. **Run Experiment:** Both groups interact with their respective versions.
  4. **Measure Outcome:** Collect and compare the results using statistical tests (usually a two-sample z-test or t-test on proportions or means).
  5. **Analyze Results:** Determine if the observed difference is statistically significant.

- **Typical Formula (Comparing Proportions):**
  
  If $p_A$ and $p_B$ are the observed success proportions in groups A and B (sample sizes $n_A$, $n_B$):

  $$
  z = \frac{p_A - p_B}{\sqrt{p(1-p)\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}
  $$

  where $p$ is the pooled probability under the null hypothesis:

  $$
  p = \frac{x_A + x_B}{n_A + n_B}
  $$

  ($x_A$ and $x_B$ are the number of successes in A and B)

- **Use Cases:**  
  - Optimizing website layout to increase conversions
  - Testing different marketing email subject lines
  - Comparing new app features against old ones for retention
  - Determining the best pricing strategies

- **Example:**  
  A company wants to know if a new sign-up button increases user sign-ups. Half of users see the old button (A), half see the new design (B). After collecting data, they compare the sign-up rates using statistical analysis to determine if the new button is significantly better.

---

**Real-life Example of A/B Testing:**

Suppose an e-commerce company wants to increase the click-through rate (CTR) on their product page. They design a new, visually distinct "Buy Now" button and want to test if it leads to more purchases.

- **Group A (Control):** Half the site visitors see the current standard "Buy Now" button.
- **Group B (Treatment):** The other half see the new, brightly colored button.

After a week, the company collects the following data:
- Group A: 5,000 visitors, 250 purchases (5% conversion rate)
- Group B: 5,000 visitors, 300 purchases (6% conversion rate)

They perform a statistical test (such as a z-test for two proportions) to see if the increase from 5% to 6% is statistically significant or just due to chance. If the difference is significant, the company can confidently switch all users to the new button design to improve sales.


### 3. Confidence Intervals

**Confidence Interval (CI):**  
A confidence interval is a range of values, derived from sample statistics, that is likely to contain the true value of an unknown population parameter (such as the mean or proportion) with a specified level of confidence (e.g., 95%).

- **What is it used for?**  
  - To estimate a population parameter (mean, proportion, etc.) based on sample data.
  - To quantify the uncertainty around an estimate.
  - To provide a range instead of a single "best guess," expressing how precise or uncertain an estimate is.

- **Formula (for the mean, normal distribution):**  
  $$
  \text{CI} = \overline{x} \pm z^* \frac{s}{\sqrt{n}}
  $$
  Where:  
  - $\overline{x}$ = sample mean  
  - $s$ = sample standard deviation  
  - $n$ = sample size  
  - $z^*$ = critical value from the standard normal distribution (1.96 for 95% confidence)

- **Use Case:**  
  Suppose you measure the heights of a random sample of 100 people and get a sample mean and standard deviation. A 95% confidence interval provides a range that likely contains the true average height for the entire population.

- **What information does it provide about the distribution?**  
  - It tells you how much your sample statistic (e.g., sample mean) is likely to vary from the actual population value due to random sampling variability.
  - A narrow CI indicates a precise estimate (less variability), while a wide CI indicates more uncertainty.
  - If you repeated the sampling process many times, about 95% of the calculated confidence intervals would contain the true parameter (for 95% confidence).
  - **Important:** The CI does *not* say there's a 95% chance the true value is in the interval (the true value is fixed)—it reflects the reliability of the estimation method.

> **Summary:**
> - Confidence intervals provide a *range* of plausible values for a population parameter.
> - They quantify sampling uncertainty and help interpret sample results in the context of the broader population.



 **What is a Confidence Interval?**

A range of values likely to contain the true population parameter, with a certain level of confidence (e.g., 95%).

**Formula for Mean (Normal Distribution)**

$$
\text{CI} = \overline{x} \pm z^* \frac{s}{\sqrt{n}}
$$

- \$ \overline{x} \$: sample mean
- \$ s \$: sample standard deviation
- \$ n \$: sample size
- \$ z^* \$: critical value (e.g., 1.96 for 95% confidence)

 **Python Example: Confidence Interval for Mean**

```python
import numpy as np
from scipy import stats

data = [112.5, 110, 115, 120, 108, 117, 113, 119, 111, 116]
mean = np.mean(data)
std_err = stats.sem(data)
conf_int = stats.t.interval(0.95, len(data)-1, loc=mean, scale=std_err)
print(f"95% Confidence Interval: {conf_int}")
```

 2. Confidence Intervals (CI)

A confidence interval gives a range likely to contain the true population mean.

**Formula: CI for Mean**

$$
\text{CI} = \overline{x} \pm t^* \frac{s}{\sqrt{n}}
$$

Where:

- \$ \overline{x} \$: sample mean
- \$ s \$: sample standard deviation
- \$ n \$: sample size
- \$ t^* \$: critical value from t-table (depends on confidence level and degrees of freedom)


**Python Example: 95% CI for Mean**

```python
import numpy as np
from scipy import stats

data = [24, 26, 23, 22, 25, 27, 28, 24, 25, 26]
mean = np.mean(data)
std_err = stats.sem(data)
conf_int = stats.t.interval(0.95, len(data)-1, loc=mean, scale=std_err)
print(f"95% Confidence Interval: {conf_int}")
```


***

### 3a. Hypothesis Testing and Confidence Intervals

Inferential statistics let you make decisions about populations using sample data. Two of the most important tools are **hypothesis testing** and **confidence intervals**. Let's break down both, with formulas and Python code examples.

***

#### 1. Hypothesis Testing

 **What is Hypothesis Testing?**

A formal process to decide if your data provides enough evidence to support a specific claim about a population parameter.[^12_2][^12_3]

#### **Key Steps**

1. **State Hypotheses**
    - Null hypothesis \$ H_0 \$: No effect or difference (e.g., "mean = 100").
    - Alternative hypothesis \$ H_1 \$: What you want to prove (e.g., "mean > 100").
2. **Choose Significance Level** \$ \alpha \$
    - Common values: 0.05, 0.01.
3. **Select the Test**
    - Depends on data type, sample size, and assumptions (e.g., t-test, z-test, ANOVA).[^12_1][^12_3]
4. **Calculate Test Statistic**
    - Example (one-sample t-test):

$$
t = \frac{\overline{x} - \mu}{s / \sqrt{n}}
$$

Where \$ \overline{x} \$ is sample mean, \$ \mu \$ is population mean, \$ s \$ is sample standard deviation, \$ n \$ is sample size.
5. **Find p-value**
    - Probability of observing your result if \$ H_0 \$ is true.
6. **Make a Decision**
    - If p-value \$ \leq \alpha \$: Reject \$ H_0 \$.
    - If p-value \$ > \alpha \$: Fail to reject \$ H_0 \$.

#### **Python Example: One-Sample t-Test**

```python
import numpy as np
from scipy import stats

data = [112.5, 110, 115, 120, 108, 117, 113, 119, 111, 116]  # sample data
population_mean = 100

# One-sample t-test
stat, p = stats.ttest_1samp(data, population_mean)
print(f"t-statistic: {stat:.2f}, p-value: {p:.4f}")

# Decision
alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis: sample mean is significantly different.")
else:
    print("Fail to reject the null hypothesis: no significant difference.")
```

**4. Quick Review**

- **Hypothesis testing**: Decide if sample data supports a claim about a population.
- **Confidence intervals**: Estimate a range for population parameters.
- **Regression**: Model and predict relationships between variables.



---

### 4. Regression Analysis

- **Definition:** Models relationships between variables (e.g., linear regression).
- **Use:** Predict outcomes, understand variable influence.

Regression Analysis (Overview)

Regression models relationships between variables. The most common is **linear regression**:

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

Where \$ Y \$ is the dependent variable, \$ X \$ is the independent variable, \$ \beta_0 \$ is the intercept, \$ \beta_1 \$ is the slope, and \$ \epsilon \$ is the error term.

**Python Example: Simple Linear Regression**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, Y)
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[^12_0]}")
```
3. Regression Modeling (Quick Overview)

Regression models relationships between variables. The most common is **linear regression**:

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

Where:

- \$ Y \$: dependent variable
- \$ X \$: independent variable
- \$ \beta_0 \$: intercept
- \$ \beta_1 \$: slope
- \$ \epsilon \$: error term


**Python Example: Simple Linear Regression**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, Y)
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[^13_0]}")
```

### 5. Correlation Analysis: Detailed Explanation

**What Is Correlation Analysis?**

Correlation analysis is a statistical method used to assess the strength and direction of the linear relationship between two quantitative variables. It helps answer: *If one variable changes, how does the other tend to change?*

- **Correlation Coefficient (\$ r \$):**
    - Ranges from **-1** to **+1**.
    - **+1:** Perfect positive correlation (as one increases, the other increases).
    - **-1:** Perfect negative correlation (as one increases, the other decreases).
    - **0:** No linear relationship.

There are different types of correlation coefficients:
- **Pearson's r:** Most common. Measures linear relationship between two continuous, normally distributed variables.
- **Spearman's rho:** Measures monotonic relationship (not necessarily linear), better for ordinal or non-normally distributed data.

---

#### Physical Significance

Correlation *quantifies* the extent to which two variables move together, but **does not imply causation**. That is, even if two variables are correlated, it doesn't mean that changes in one *cause* changes in the other.

- **Magnitude** (|r|): Strength of relation; closer to 1 means stronger relationship.
- **Sign** (±): Direction of relationship; positive or negative.

A **strong correlation** suggests a predictable relationship, which can be useful in modeling, forecasting, cluster detection, and more.

---

#### Use Cases

- **Feature Selection:** In machine learning/data science, correlation is used to identify variables that contribute similar (redundant) information, helping to remove or combine features.
- **Exploratory Data Analysis (EDA):** Understand relationships before modeling (e.g., are two measurements highly correlated?).
- **Finance:** Measure how two stocks move together (portfolio diversification).
- **Medical Research:** Find out if a biomarker is associated with a disease outcome.

---

#### Real-Life Examples

1. **Height and Weight:**
    - Taller people tend to weigh more. There is typically a strong positive correlation between height and weight in human populations.

2. **Hours Studied vs. Exam Score:**
    - Students who study more hours tend to score higher, showing a positive correlation.

3. **Ice Cream Sales and Temperature:**
    - On hotter days, more ice creams are sold. Positive correlation.

4. **Number of Firefighters and Size of Fire:**
    - More firefighters are present at larger fires (positive correlation), but increasing firefighters doesn't cause the fire to grow (shows why "correlation is not causation").

---

#### Example: Calculating Pearson Correlation in Python

```python
import numpy as np
from scipy.stats import pearsonr

# Example data: hours studied and exam scores
hours_studied = [2, 3, 5, 6, 8, 10, 12]
exam_score = [65, 70, 80, 88, 95, 98, 100]

# Calculate Pearson correlation coefficient and p-value
r_value, p_value = pearsonr(hours_studied, exam_score)
print(f"Pearson correlation coefficient: {r_value:.2f}")
print(f"P-value: {p_value:.4f}")
```

**Interpretation:**
- If \$ r = 0.97 \$, this indicates a strong positive correlation: as hours studied increases, so do exam scores.
- A small p-value (e.g., < 0.05) suggests the correlation is statistically significant.

---

**Summary Table: Levels of Correlation**

| r value      | Strength of correlation      |
|--------------|-----------------------------|
| 0.9 to 1.0   | Very strong                 |
| 0.7 to 0.9   | Strong                      |
| 0.5 to 0.7   | Moderate                    |
| 0.3 to 0.5   | Weak                        |
| 0.0 to 0.3   | Very weak or none           |

*Remember:* Correlation measures *association*, not *causation*.


- **Definition:** Measures strength and direction of relationship between two variables.
- **Coefficient:** Pearson’s $r$ ranges from -1 to 1.

***








##**Types of Data**

- **Qualitative (Categorical):** Descriptive, non-numeric (e.g., colors, labels).
- **Quantitative (Numerical):** Numeric values, can be discrete (countable) or continuous (measurable).[^7_2]
---

### 6. What is the t-Test Value?

1. **Definition of t-Test Value**

The **t-test value** (or t-statistic) is a number calculated during a t-test that measures how far your observed difference (between means) is from what you’d expect under the null hypothesis, relative to the variability in your data.[^21_1][^21_5][^21_6]

***What is a t-Test?**

Let's focus on the **t-test** (the most common hypothesis test for means), how to calculate **confidence intervals**, and a quick overview of **regression modeling**. These are foundational tools for data scientists.

A t-test helps you determine if the means of two groups are statistically different. It's used when:

- Your data is continuous and (ideally) normally distributed.
- The population standard deviation is unknown.
- Sample sizes are moderate (often < 30), but t-test is robust for larger samples too.


**Types of t-Tests**

- **One-sample t-test:** Compare sample mean to a known value.
- **Two-sample t-test:** Compare means of two independent groups.
- **Paired t-test:** Compare means from the same group at different times.


**Formula: Two-Sample t-Test**

For two independent samples:

$$
t = \frac{\overline{x}_1 - \overline{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$

Where:

- \$ \overline{x}_1, \overline{x}_2 \$: sample means
- \$ s_1^2, s_2^2 \$: sample variances
- \$ n_1, n_2 \$: sample sizes

A larger t-value means a more significant difference between groups.[^13_1][^13_6]

 **Python Example: Two-Sample t-Test**

```python
import numpy as np
from scipy import stats

group1 = [24, 26, 23, 22, 25, 27, 28, 24, 25, 26]
group2 = [20, 19, 21, 22, 20, 18, 19, 21, 20, 19]

# Two-sample t-test (assume equal variances)
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
```

- If \$ p \leq \alpha \$ (e.g., 0.05), reject the null hypothesis: groups are significantly different.


**Summary Table**

| Test/Method | Purpose | Formula/Code Example |
| :-- | :-- | :-- |
| t-Test | Compare means | `stats.ttest_ind()` |
| Confidence Interval | Estimate range for mean | `stats.t.interval()` |
| Regression | Model/predict relationships | `LinearRegression().fit()` |


2. **How is it Calculated?**

For a two-sample t-test:

$$
t = \frac{\overline{x}_1 - \overline{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$

- \$ \overline{x}_1, \overline{x}_2 \$: sample means
- \$ s_1^2, s_2^2 \$: sample variances
- \$ n_1, n_2 \$: sample sizes

3. **Interpreting the t-Test Value**

- The **t-value** tells you how many standard errors your observed difference is away from zero (no difference).
- **Large absolute t-value**: Strong evidence against the null hypothesis (means are likely different).
- **Small t-value**: Weak evidence; observed difference could be due to random chance.[^21_2][^21_5]


4. **Relation to p-Value**

- The t-value is used to calculate the **p-value**.
- The p-value tells you the probability of seeing a t-value as extreme as yours if the null hypothesis is true.
- If the t-value is large (in absolute terms), the p-value will be small, indicating statistical significance.[^21_5][^21_1]


5. **Example in Sales Scenario**

Suppose you run a t-test comparing sales with and without advertisement and get a t-value of 2.5:

- This means the difference in sales is 2.5 standard errors away from zero.
- If your degrees of freedom are 38, a t-value of 2.5 corresponds to a p-value of about 0.017.
- Since 0.017 < 0.05, you reject the null hypothesis and conclude advertising had a significant effect.[^21_1][^21_5]


6. **Summary Table**

| Statistic | What it Means | Interpretation |
| :-- | :-- | :-- |
| t-value | Standardized difference between means | Large (absolute) t-value = strong evidence against null hypothesis |
| p-value | Probability of observing such a t-value if null is true | Small p-value (< 0.05) = statistically significant difference |


now provide me a complete code for the t-test and p-value and hypothesis test for sales scenario

Here is a complete Python code example for performing a two-sample t-test to compare sales with and without advertisement, including calculation of the t-statistic, p-value, and hypothesis test interpretation:[^22_2][^22_6][^22_8]

```python
import numpy as np
from scipy.stats import ttest_ind

# Example sales data (weekly sales)
# Group 1: Sales without advertisement
sales_no_ad = np.array([200, 220, 210, 205, 215, 225, 210, 208, 212, 218])

# Group 2: Sales with advertisement
sales_with_ad = np.array([230, 240, 235, 245, 238, 250, 242, 239, 241, 248])

# Step 1: Calculate means for reference
mean_no_ad = np.mean(sales_no_ad)
mean_with_ad = np.mean(sales_with_ad)
print(f"Mean sales without ad: {mean_no_ad:.2f}")
print(f"Mean sales with ad: {mean_with_ad:.2f}")

# Step 2: Perform two-sample t-test
t_stat, p_value = ttest_ind(sales_with_ad, sales_no_ad, equal_var=True)
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.4f}")

# Step 3: Interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Advertising significantly increased sales.")
else:
    print("Fail to reject the null hypothesis: No significant difference in sales due to advertising.")

# Optional: Print difference in means
print(f"Difference in means: {mean_with_ad - mean_no_ad:.2f}")
```

**How to interpret:**

- If the p-value is less than 0.05, you conclude that advertising had a statistically significant effect on sales.
- The t-statistic quantifies how far apart the group means are, relative to the variability in the data.


---