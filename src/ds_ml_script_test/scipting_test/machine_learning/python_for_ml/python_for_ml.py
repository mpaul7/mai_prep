import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Step 1: Create a synthetic CSV file
# -------------------------
data = {
    'feature1': [5.1, 4.9, 4.7, np.nan, 5.0, 5.4, 4.6, 5.0, 5.2, 4.8],
    'feature2': [3.5, np.nan, 3.2, 3.1, 3.6, 3.9, 3.4, 3.7, 3.8, 3.0],
    'category': ['Red', 'Blue', 'Red', 'Green', 'Blue', 'Red', 'Blue', 'Green', 'Red', 'Blue'],
    'target': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
df.to_csv('synthetic_data.csv', index=False)
print("Synthetic CSV file created: 'synthetic_data.csv'\n")

# -------------------------
# Step 2: Read CSV
# -------------------------
df = pd.read_csv('synthetic_data.csv')
print("Initial DataFrame:\n", df.head(), "\n")

# -------------------------
# Step 3: Handle missing values
# Numeric columns → fill with mean
# -------------------------
for col in ['feature1', 'feature2']:
    df[col] = df[col].fillna(df[col].mean())

# -------------------------
# Step 4: Encode categorical column (One-hot)
# -------------------------
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)

# -------------------------
# Step 5: Feature Engineering
# New column = sum of features
# -------------------------
df_encoded['feature_sum'] = df_encoded['feature1'] + df_encoded['feature2']

# Groupby example: mean of feature1 by target
grouped = df_encoded.groupby('target')['feature1'].mean()
print("Groupby mean of feature1 by target:\n", grouped, "\n")

# -------------------------
# Step 6: NumPy Vectorized Operations
# -------------------------
X = df_encoded.drop('target', axis=1).values
y = df_encoded['target'].values

# Example: square all features
X_squared = X ** 2
print("First row of squared features:", X_squared[0], "\n")

# -------------------------
# Step 7: Plotting
# -------------------------
# Histogram
plt.hist(df_encoded['feature1'], bins=5, color='skyblue', edgecolor='black')
plt.title("Feature1 Distribution")
plt.xlabel("Feature1")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.boxplot(df_encoded['feature2'])
plt.title("Feature2 Boxplot")
plt.show()

# Scatter plot
plt.scatter(df_encoded['feature1'], df_encoded['feature2'], c=df_encoded['target'], cmap='viridis')
plt.title("Feature1 vs Feature2")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.show()

# Correlation heatmap
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
