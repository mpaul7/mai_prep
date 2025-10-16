import numpy as np

# Example dataset: numeric + categorical + missing values
X = [
    [5.1, 3.5, 'Red'],
    [4.9, None, 'Blue'],
    [4.7, 3.2, 'Red'],
    [None, 3.1, 'Green'],
    [5.0, 3.6, 'Blue'],
    [5.4, 3.9, 'Red'],
    [4.6, 3.4, None]
]

y = [0, 1, 0, 1, 0, 0, 1]  # target variable

# -------------------------
# Step 1: Convert to numpy array
# -------------------------
X_array = np.array(X, dtype=object)
y_array = np.array(y, dtype=float)

# -------------------------
# Step 2: Handle missing values (fill with column mean for numeric, mode for categorical)
# -------------------------
n_rows, n_cols = X_array.shape

for j in range(n_cols):
    col = X_array[:, j]
    is_numeric = all(isinstance(v, (int, float, type(None))) or str(v).replace('.', '', 1).isdigit() for v in col)
    
    if is_numeric:
        col_numeric = np.array([float(v) if v is not None else np.nan for v in col], dtype=float)
        mean_val = np.nanmean(col_numeric)
        col_numeric[np.isnan(col_numeric)] = mean_val
        X_array[:, j] = col_numeric
    else:
        non_missing = [v for v in col if v is not None]
        mode_val = max(set(non_missing), key=non_missing.count)
        col_filled = [v if v is not None else mode_val for v in col]
        X_array[:, j] = col_filled

# Convert numeric columns to float
for j in range(n_cols):
    if all(isinstance(v, (int, float, np.number)) or str(v).replace('.', '', 1).isdigit() for v in X_array[:, j]):
        X_array[:, j] = X_array[:, j].astype(float)

# -------------------------
# Step 3: Encode categorical columns (one-hot encoding)
# -------------------------
encoded_cols = []
for j in range(n_cols):
    if not np.issubdtype(X_array[:, j].dtype, np.number):
        unique_vals = sorted(list(set(X_array[:, j])))
        one_hot = np.zeros((n_rows, len(unique_vals)))
        for i, val in enumerate(X_array[:, j]):
            one_hot[i, unique_vals.index(val)] = 1
        for k in range(one_hot.shape[1]):
            encoded_cols.append(one_hot[:, k])
    else:
        encoded_cols.append(X_array[:, j].astype(float))

X_encoded = np.column_stack(encoded_cols)

# -------------------------
# Step 4: Scaling / normalization (Min-Max)
# -------------------------
X_scaled = np.zeros_like(X_encoded, dtype=float)
for j in range(X_encoded.shape[1]):
    col = X_encoded[:, j].astype(float)
    min_val = np.min(col)
    max_val = np.max(col)
    X_scaled[:, j] = (col - min_val) / (max_val - min_val) if max_val != min_val else 0.0

# -------------------------
# Step 5: Outlier removal (Z-score > 3)
# -------------------------
z_scores = (X_scaled - np.mean(X_scaled, axis=0)) / np.std(X_scaled, axis=0)
rows_to_keep = ~np.any(np.abs(z_scores) > 3, axis=1)
X_scaled_no_outliers = X_scaled[rows_to_keep]
y_no_outliers = y_array[rows_to_keep]

# -------------------------
# Step 6: Train-test split (70-30)
# -------------------------
n_samples = X_scaled_no_outliers.shape[0]
indices = np.arange(n_samples)
np.random.shuffle(indices)

train_size = int(0.7 * n_samples)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = X_scaled_no_outliers[train_idx]
X_test = X_scaled_no_outliers[test_idx]
y_train = y_no_outliers[train_idx]
y_test = y_no_outliers[test_idx]

# -------------------------
# Step 7: Feature selection
# Variance threshold (<0.01) and correlation (>0.9)
# -------------------------
variances = np.var(X_train, axis=0)
keep_idx = [i for i, v in enumerate(variances) if v >= 0.01]
X_train_fs = X_train[:, keep_idx]
X_test_fs = X_test[:, keep_idx]

corr_matrix = np.corrcoef(X_train_fs, rowvar=False)
to_remove = set()
threshold = 0.9
n_features_fs = X_train_fs.shape[1]
for i in range(n_features_fs):
    for j in range(i+1, n_features_fs):
        if abs(corr_matrix[i, j]) > threshold:
            to_remove.add(j)

final_idx = [i for i in range(X_train_fs.shape[1]) if i not in to_remove]
X_train_final = X_train_fs[:, final_idx]
X_test_final = X_test_fs[:, final_idx]

# -------------------------
# Print final processed data
# -------------------------
print("X_train_final:\n", X_train_final)
print("X_test_final:\n", X_test_final)
print("y_train:", y_train)
print("y_test:", y_test)
