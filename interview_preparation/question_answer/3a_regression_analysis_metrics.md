--

# Regression Model Evaluation 
## Model Evaluation Metrics

**For Regression:**

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared ($R^2$)
- Adjusted R-squared

**For Classification (preview for next chapters):**

- Accuracy
- Precision, Recall, F1 Score
- ROC-AUC
- Confusion Matrix


## Regression Metrics

Regression metrics help you evaluate how well your model predicts continuous outcomes. They quantify prediction errors and model fit, guiding both technical improvements and business decisions.



**A. Mean Absolute Error (MAE)**

- **Formula:**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$
- **Python:**

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
```

- **Interpretation:** Average absolute difference between actual and predicted values. Lower is better.
- **Business Significance:** Directly tells you the typical error in your forecasts (e.g., average sales prediction error in dollars).

***

**B. Mean Squared Error (MSE)**

- **Formula:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$
- **Python:**

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
```

- **Interpretation:** Average squared difference between actual and predicted values. Penalizes larger errors more than MAE.
- **Business Significance:** Useful for risk assessment; highlights large mistakes in predictions.

***

**C. Root Mean Squared Error (RMSE)**

- **Formula:**

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$
- **Python:**

```python
import numpy as np
rmse = np.sqrt(mse)
print("RMSE:", rmse)
# Or directly:
rmse = mean_squared_error(y_test, y_pred, squared=False)
```

- **Interpretation:** Typical error in the same units as the target variable. Lower is better.
- **Business Significance:** Easy to communicate; shows how far off predictions are, on average.

**D. R-squared ($R^2$)**

- **Formula:**

$$
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \overline{y})^2}
$$
- **Python:**

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)
```

- **Interpretation:** Proportion of variance in the target explained by the model. Ranges from 0 to 1 (higher is better).
- **Business Significance:** Indicates how much of the outcome your model can explain; high $R^2$ means reliable predictions for planning and strategy.


**E. Adjusted R-squared**

- **Formula:**

$$
R^2_{adj} = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)
$$

Where $n$ is sample size, $p$ is number of predictors.
- **Python:** Not directly in sklearn, but can be calculated:

```python
n = len(y_test)
p = X_test.shape[^36_1]
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R-squared:", r2_adj)
```

- **Interpretation:** Penalizes for adding irrelevant features; more honest measure of model fit.
- **Business Significance:** Helps avoid overfitting when comparing models with different numbers of predictors.

***

**Complete Python Example**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Example data
X_train = np.array([[^36_10], [^36_12], [^36_15], [^36_18], [^36_20], [^36_22], [^36_25], [^36_28], [^36_30], [^36_32]])
y_train = np.array([200, 210, 220, 230, 240, 250, 265, 270, 280, 290])
X_test = np.array([[^36_14], [^36_16], [^36_19], [^36_21], [^36_24], [^36_27], [^36_29], [^36_31], [^36_33], [^36_35]])
y_test = np.array([215, 225, 235, 245, 255, 265, 275, 285, 295, 305])

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
```



**Business Significance: Quick Reference Table**

| Metric | What It Means in Business | How to Use It |
| :-- | :-- | :-- |
| MAE | Avg. prediction error (units) | Budgeting, risk assessment |
| MSE | Penalizes large errors | Spot big mistakes |
| RMSE | Typical error (units) | Communicate forecast error |
| R-squared | % of outcome explained by model | Trustworthiness of forecasts |
| Adjusted R² | Honest fit with many features | Feature selection, overfit |



## Physical Significance: Regression Metrics with Practical Business Examples

Regression metrics are not just numbers—they translate directly into business impact. Let’s explore how to interpret these metrics using real-world scenarios, with concrete values and percentage changes to illustrate their significance.

***

1. **Mean Absolute Error (MAE)**

- **Definition:** Average absolute difference between predicted and actual values.
- **Business Example:**
    - *Sales Forecasting*: Suppose your model predicts weekly sales for a retail store. If MAE = \$2,000, on average, your predictions are off by \$2,000 per week.
    - *Impact*: If your weekly sales are typically \$20,000, this is a 10% error. If your business can tolerate a 10% deviation, the model is actionable. If not, you need to improve accuracy.

2. **Root Mean Squared Error (RMSE)**

- **Definition:** Square root of the average squared error; penalizes larger mistakes.
- **Business Example:**
    - *Real Estate Pricing*: Your model predicts house prices. RMSE = \$15,000, and the average house price is \$300,000. This means typical prediction error is 5% of the sale price.
    - *Impact*: If your business needs to price homes within \$10,000, this model may need improvement. RMSE helps you set realistic expectations for pricing accuracy.

3. **R-squared ($R^2$)**

- **Definition:** Proportion of variance in the target explained by the model.
- **Business Example:**
    - *Marketing ROI*: You model the impact of ad spend on sales. If $R^2 = 0.85$, then 85% of sales variation is explained by ad spend and other predictors.
    - *Impact*: High $R^2$ means your model is reliable for forecasting and resource allocation. If $R^2$ is low (e.g., 0.40), most sales variation is unexplained, so forecasts are less trustworthy.

4. **Adjusted R-squared**

- **Definition:** Adjusts $R^2$ for the number of predictors, penalizing irrelevant features.
- **Business Example:**
    - *Customer Satisfaction*: You model Net Promoter Score (NPS) using wait time, price, and product quality. If adding more features doesn’t increase adjusted $R^2$, those features don’t help explain NPS and may be dropped.
    - *Impact*: Ensures your model is not overfitting by adding unnecessary variables.

5. **Interpreting Metrics for Business Decisions**

**A. Good Model (High $R^2$, Low MAE/RMSE)**

- *Example*: An e-commerce company predicts weekly sales with $R^2 = 0.90$, MAE = \$1,000, RMSE = \$1,500. Weekly sales average \$20,000.
- *Business Impact*: Model explains 90% of sales variation; typical error is 5–7.5%. The company can confidently use forecasts for inventory planning and marketing spend.


**B. Poor Model (Low $R^2$, High MAE/RMSE)**

- *Example*: Same company, but $R^2 = 0.40$, MAE = \$4,000, RMSE = \$6,000.
- *Business Impact*: Model explains only 40% of sales variation; errors are 20–30%. Forecasts are unreliable—company should improve the model by adding features (e.g., seasonality, promotions) or using more advanced algorithms.

***

6. **Concrete Use Cases and Actions**

- **Price Elasticity**: Regression shows that a \$1 increase in price reduces sales by 50 units. If RMSE is low, you can confidently adjust prices to optimize revenue.
- **Customer Satisfaction**: Regression finds that reducing wait time by 5 minutes increases NPS by 10 points. If $R^2$ is high, investing in faster service is justified.
- **Marketing Effectiveness**: Regression reveals that every \$1,000 in YouTube ads increases sales by \$48,000. If MAE is low, you can forecast ROI and allocate budget efficiently.[^37_7]

***

7. **Summary Table: Metrics and Business Impact**

| Metric | Example Value | Business Interpretation | Actionable Decision |
| :-- | :-- | :-- | :-- |
| MAE | \$2,000 | Avg. error in sales prediction (10%) | Acceptable for planning if <10% |
| RMSE | \$1,500 | Typical error (7.5%) | Use for risk assessment |
| R-squared | 0.85 | 85% of sales explained by model | Reliable for forecasting |
| Adj. R-squared | 0.83 | Model not overfitting | Keep only useful features |


***

8. **Key Takeaways for Interviews and Practice**

- Always relate metrics to business context: What does a \$2,000 error mean for your company?
- Use percentage errors to communicate impact to stakeholders.
- High $R^2$ and low errors mean your model is ready for real-world decisions; low values mean you need to improve features, data, or algorithms.
- Adjusted $R^2$ helps prevent overfitting and keeps your model interpretable.