

---

# Non-parametric Methods

## 10.1 **Definition**

Non-parametric methods are statistical techniques that do **not** assume a specific probability distribution for the data (such as normality). They are especially useful when:

- The data is ordinal or ranked (not interval/ratio scale).
- The sample size is small.
- The data is skewed, contains outliers, or violates assumptions required for parametric tests.


## 10.2 **Common Non-parametric Tests**

- **Mann-Whitney U Test:** Compares medians of two independent groups (alternative to the independent t-test).
- **Wilcoxon Signed-Rank Test:** Compares medians of two related groups (alternative to the paired t-test).
- **Kruskal-Wallis Test:** Compares medians across more than two groups (alternative to one-way ANOVA).
- **Friedman Test:** Compares more than two related groups (alternative to repeated measures ANOVA).
- **Chi-Square Test:** Tests for association between categorical variables.
- **Kolmogorov-Smirnov Test:** Tests if a sample comes from a specific distribution.


## 10.3 **Business Impact**

- **Robustness:** Non-parametric methods provide reliable results when data is non-normal, contains outliers, or is ordinal.
- **Flexibility:** Useful in market research (e.g., customer satisfaction surveys with ranked responses), healthcare (e.g., comparing treatment effects with non-normal outcomes), and finance (e.g., comparing returns with heavy tails).
- **Risk Reduction:** Avoids misleading conclusions that can arise from using parametric tests on inappropriate data.


## 10.4 **Practical Examples**

### **A. Mann-Whitney U Test**

- **Scenario:** A company wants to compare customer satisfaction scores (on a 1–5 scale) between two products, but the scores are not normally distributed.
- **Python Example:**

```python
from scipy.stats import mannwhitneyu
group1 = [4, 5, 3, 4, 4]
group2 = [2, 3, 2, 3, 1]
stat, p = mannwhitneyu(group1, group2)
print(f"Mann-Whitney U p-value: {p:.3f}")
```


### **B. Wilcoxon Signed-Rank Test**

- **Scenario:** A hospital tests a new drug on patients and measures their blood pressure before and after treatment (paired data, not normal).
- **Python Example:**

```python
from scipy.stats import wilcoxon
before = [140, 138, 150, 148, 142]
after = [135, 136, 145, 144, 140]
stat, p = wilcoxon(before, after)
print(f"Wilcoxon p-value: {p:.3f}")
```


### **C. Kruskal-Wallis Test**

- **Scenario:** Comparing customer satisfaction across three stores with non-normal data.
- **Python Example:**

```python
from scipy.stats import kruskal
store1 = [4, 5, 4, 4]
store2 = [3, 3, 2, 4]
store3 = [5, 4, 5, 5]
stat, p = kruskal(store1, store2, store3)
print(f"Kruskal-Wallis p-value: {p:.3f}")
```


## 10.5 **Summary Table: Non-parametric Methods**

| Test/Method | Use Case | Business Example |
| :-- | :-- | :-- |
| Mann-Whitney U | 2 independent groups | Product A vs. B satisfaction |
| Wilcoxon Signed-Rank | 2 related groups | Before/after treatment |
| Kruskal-Wallis | >2 independent groups | Multiple store comparison |
| Chi-Square | Categorical association | Market segment vs. purchase |
| Kolmogorov-Smirnov | Distribution comparison | Model fit to observed data |


***

**Key Takeaway:**
Non-parametric methods are essential tools for robust, assumption-free analysis—especially when data is non-normal, ordinal, or contains outliers. They help businesses make sound decisions without relying on strict statistical assumptions.


