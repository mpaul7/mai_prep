import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Create synthetic numeric data
# -----------------------------
np.random.seed(42)

# Normal data + outliers
salaries = np.concatenate([
    np.random.randint(30000, 60000, 20),
    np.array([150000, 200000, 1000000])  # outliers
])

print("Original salaries:\n", salaries)
print("\nOriginal mean:", np.mean(salaries))
print("Original std:", np.std(salaries))
print("Original length:", len(salaries), "\n")

# Plot before removal
plt.boxplot(salaries, patch_artist=True)
plt.title("Before Outlier Removal")
plt.ylabel("Salary")
plt.show()

# -----------------------------
# Step 2: IQR Method
# -----------------------------
Q1 = np.percentile(salaries, 25)
Q3 = np.percentile(salaries, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mask_iqr = (salaries >= lower_bound) & (salaries <= upper_bound)
salaries_iqr_cleaned = salaries[mask_iqr]

print("After IQR Removal:")
print("Count:", len(salaries_iqr_cleaned))
print("Mean:", np.mean(salaries_iqr_cleaned))
print("Std:", np.std(salaries_iqr_cleaned), "\n")

plt.boxplot(salaries_iqr_cleaned, patch_artist=True)
plt.title("After IQR Outlier Removal")
plt.ylabel("Salary")
plt.show()

# -----------------------------
# Step 3: Z-Score Method
# -----------------------------
mean = np.mean(salaries)
std = np.std(salaries)
z_scores = (salaries - mean) / std

mask_z = np.abs(z_scores) <= 3
salaries_z_cleaned = salaries[mask_z]

print("After Z-Score Removal:")
print("Count:", len(salaries_z_cleaned))
print("Mean:", np.mean(salaries_z_cleaned))
print("Std:", np.std(salaries_z_cleaned), "\n")

plt.boxplot(salaries_z_cleaned, patch_artist=True)
plt.title("After Z-Score Outlier Removal")
plt.ylabel("Salary")
plt.show()

# -----------------------------
# Step 4: Percentile Capping
# -----------------------------
lower_cap = np.percentile(salaries, 5)
upper_cap = np.percentile(salaries, 95)

salaries_capped = np.clip(salaries, lower_cap, upper_cap)

print("After Percentile Capping:")
print("Count:", len(salaries_capped))
print("Mean:", np.mean(salaries_capped))
print("Std:", np.std(salaries_capped), "\n")

plt.boxplot(salaries_capped, patch_artist=True)
plt.title("After Percentile Capping")
plt.ylabel("Salary")
plt.show()

