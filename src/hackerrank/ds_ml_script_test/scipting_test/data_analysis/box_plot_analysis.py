import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Step 1: Create synthetic dataset
# -------------------------
np.random.seed(42)  # for reproducibility

data = {
    'Age': np.random.randint(18, 60, 30),                 # Numeric feature
    'Salary': np.random.randint(30000, 120000, 30),       # Numeric feature
    'Department': np.random.choice(['HR', 'IT', 'Sales'], 30),  # Categorical
}

df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")

# -------------------------
# Step 2: Boxplot Analysis for Outliers
# -------------------------
# Boxplot of Salary
plt.figure(figsize=(6, 4))
plt.boxplot(df['Salary'], patch_artist=True)
plt.title("Salary Distribution Boxplot")
plt.ylabel("Salary")
plt.show()

# -------------------------
# Step 3: Detect outliers using IQR
# -------------------------
Q1 = np.percentile(df['Salary'], 25)
Q3 = np.percentile(df['Salary'], 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
print("Detected Outliers in Salary:\n", outliers, "\n")

# -------------------------
# Step 4: Boxplot by Department
# -------------------------
departments = df['Department'].unique()
salary_by_dept = [df[df['Department']==dept]['Salary'] for dept in departments]
print("Salary by Department:\n", salary_by_dept, "\n")

plt.figure(figsize=(8, 5))
plt.boxplot(salary_by_dept, labels=departments, patch_artist=True)
plt.title("Salary Distribution by Department")
plt.ylabel("Salary")
plt.show()

# -------------------------
# Step 5: Basic Insights
# -------------------------
for dept in departments:
    dept_salaries = df[df['Department']==dept]['Salary']
    print(f"{dept} -> Min: {dept_salaries.min()}, Q1: {np.percentile(dept_salaries, 25)}, Median: {np.median(dept_salaries)}, Q3: {np.percentile(dept_salaries,75)}, Max: {dept_salaries.max()}")
