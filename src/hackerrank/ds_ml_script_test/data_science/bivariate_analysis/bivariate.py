import numpy as np
import pandas as pd

def linear_regression_manual():
    # ============================================================
    # 1️⃣ Create sample dataset with missing + categorical values
    # ============================================================
    data = {
        'age': [25, 30, np.nan, 22, 28, 35, np.nan, 40],
        'experience': [1, 3, 5, 2, np.nan, 10, 7, np.nan],
        'city': ['NY', 'LA', 'NY', 'SF', 'LA', 'SF', 'NY', 'LA'],
        'salary': [45000, 50000, 60000, 48000, 52000, 80000, 75000, 70000]
    }
    df = pd.DataFrame(data)

    # ============================================================
    # 2️⃣ Handle missing values (impute with mean)
    # ============================================================
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # ============================================================
    # 3️⃣ Encode categorical feature (One-hot encoding)
    # ============================================================
    df = pd.get_dummies(df, columns=['city'], drop_first=True)

    # ============================================================
    # 4️⃣ Normalize numeric columns (optional, for stability)
    # ============================================================
    for col in df.columns:
        if col != 'salary':
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # ============================================================
    # 5️⃣ Choose one feature (e.g., age) for simple linear regression
    # ============================================================
    x = df['age'].values
    y = df['salary'].values

    # ============================================================
    # 6️⃣ Manual Linear Regression (y = m*x + b)
    # ============================================================
    x_array = np.array(x)
    y_array = np.array(y)

    n = len(x_array)
    x_mean = np.mean(x_array)
    y_mean = np.mean(y_array)

    numerator = np.sum((x_array - x_mean) * (y_array - y_mean))
    denominator = np.sum((x_array - x_mean) ** 2)

    if denominator == 0:
        raise ValueError("Cannot perform regression: no variation in x variable")

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # ============================================================
    # 7️⃣ Predictions
    # ============================================================
    predictions = slope * x_array + intercept

    # ============================================================
    # 8️⃣ Calculate metrics (MAE, MSE, RMSE, R²)
    # ============================================================
    mae = np.mean(np.abs(y_array - predictions))
    mse = np.mean((y_array - predictions) ** 2)
    rmse = np.sqrt(mse)
    ss_total = np.sum((y_array - np.mean(y_array)) ** 2)
    ss_residual = np.sum((y_array - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # ============================================================
    # 9️⃣ Display Results
    # ============================================================
    print("===== Manual Linear Regression (One Feature) =====")
    print("\nCleaned Data:\n", df.head())
    print(f"\nFeature Used: age")
    print(f"Slope (m): {slope:.4f}")
    print(f"Intercept (b): {intercept:.4f}")

    print("\n===== Model Evaluation =====")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

# Run the function
linear_regression_manual()
