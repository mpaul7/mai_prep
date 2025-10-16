import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Step 1: Create synthetic dataset
# -------------------------
np.random.seed(42)
data = {
    'Age': np.random.randint(18, 60, 20),
    'Salary': [30000, 32000, 31000, 30500, 400000, 45000, 47000, 48000, 49000, 50000,
               51000, 52000, 53000, 54000, 55000, 1000000, 57000, 58000, 59000, 60000]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df, "\n")

# -------------------------
# Step 2: Boxplot / IQR method
# -------------------------
Q1 = np.percentile(df['Salary'], 25)
Q3 = np.percentile(df['Salary'], 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_iqr = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
print("Outliers detected by IQR method:\n", outliers_iqr, "\n")

# Remove outliers
df_iqr_cleaned = df[(df['Salary'] >= lower_bound) & (df['Salary'] <= upper_bound)]
print("Dataset after IQR outlier removal:\n", df_iqr_cleaned, "\n")

# Plot
plt.boxplot(df['Salary'], patch_artist=True)
plt.title("Salary Distribution (Before IQR Outlier Removal)")
plt.show()

plt.boxplot(df_iqr_cleaned['Salary'], patch_artist=True)
plt.title("Salary Distribution (After IQR Outlier Removal)")
plt.show()

# -------------------------
# Step 3: Z-score method
# -------------------------
mean_salary = np.mean(df['Salary'])
std_salary = np.std(df['Salary'])
z_scores = (df['Salary'] - mean_salary) / std_salary
print("Z-scores:\n", z_scores, "\n")

# Identify outliers |z| > 3
outliers_z = df[np.abs(z_scores) > 3]
print(f"outliers_z: {outliers_z}")
print("Outliers detected by Z-score method:\n", outliers_z, "\n")

# Remove outliers
df_z_cleaned = df[np.abs(z_scores) <= 3]
print("Dataset after Z-score outlier removal:\n", df_z_cleaned, "\n")

# -------------------------
# Step 4: Percentile Capping / Trimming
# -------------------------
lower_percentile = np.percentile(df['Salary'], 5)
upper_percentile = np.percentile(df['Salary'], 95)

# Cap salaries
df_percentile_capped = df.copy()
df_percentile_capped['Salary'] = np.clip(df['Salary'], lower_percentile, upper_percentile)
print("Dataset after percentile capping:\n", df_percentile_capped, "\n")

# Plot after capping
plt.boxplot(df_percentile_capped['Salary'], patch_artist=True)
plt.title("Salary Distribution (After Percentile Capping)")
plt.show()
