"""
Pandas Common Operations - Interview Preparation
Topic 9.3: Common Operations

This module covers:
- Data Cleaning: Handling missing values (basics)
- Aggregations: sum(), mean(), count(), groupby() basics
- Merging: Basic join operations
"""

import pandas as pd
import numpy as np

# ============================================================================
# 1. DATA CLEANING - HANDLING MISSING VALUES
# ============================================================================

print("=" * 70)
print("1. DATA CLEANING - HANDLING MISSING VALUES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Detecting Missing Values
# ----------------------------------------------------------------------------
print("\n--- 1.1 Detecting Missing Values ---")

# Create DataFrame with missing values
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, None, 35, 40, None],
    'salary': [50000, 60000, None, 80000, 55000],
    'city': ['New York', 'London', None, 'Paris', 'Berlin']
})
print(f"DataFrame with missing values:\n{df}")

# Check for missing values
print(f"\nisna():\n{df.isna()}")
print(f"\nisnull():\n{df.isnull()}")  # Same as isna()
print(f"\nnotna():\n{df.notna()}")

# Count missing values per column
print(f"\nMissing values per column:\n{df.isna().sum()}")

# Count missing values total
print(f"\nTotal missing values: {df.isna().sum().sum()}")

# Percentage of missing values
print(f"\nPercentage missing:\n{(df.isna().sum() / len(df) * 100).round(2)}")


# ----------------------------------------------------------------------------
# 1.2 Dropping Missing Values
# ----------------------------------------------------------------------------
print("\n--- 1.2 Dropping Missing Values ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, None, 35, None],
    'salary': [50000, 60000, None, 80000]
})
print(f"Original DataFrame:\n{df}")

# Drop rows with any missing values
print(f"\ndropna() - drop rows with any NaN:\n{df.dropna()}")

# Drop rows where all values are missing
print(f"\ndropna(how='all') - drop rows where all NaN:\n{df.dropna(how='all')}")

# Drop columns with any missing values
print(f"\ndropna(axis=1) - drop columns with any NaN:\n{df.dropna(axis=1)}")

# Drop rows with missing values in specific columns
print(f"\ndropna(subset=['age']) - drop rows with NaN in 'age':\n{df.dropna(subset=['age'])}")

# Drop rows with missing values in multiple columns
print(f"\ndropna(subset=['age', 'salary']):\n{df.dropna(subset=['age', 'salary'])}")


# ----------------------------------------------------------------------------
# 1.3 Filling Missing Values
# ----------------------------------------------------------------------------
print("\n--- 1.3 Filling Missing Values ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, None, 35, None],
    'salary': [50000, 60000, None, 80000],
    'city': ['New York', None, 'Tokyo', 'Paris']
})
print(f"Original DataFrame:\n{df}")

# Fill with specific value
print(f"\nfillna(0) - fill all NaN with 0:\n{df.fillna(0)}")

# Fill with different values per column
print(f"\nfillna({{'age': 0, 'salary': 0, 'city': 'Unknown'}}):\n{df.fillna({'age': 0, 'salary': 0, 'city': 'Unknown'})}")

# Forward fill (ffill) - use previous value
print(f"\nffill() - forward fill:\n{df.ffill()}")

# Backward fill (bfill) - use next value
print(f"\nbfill() - backward fill:\n{df.bfill()}")

# Fill with mean (for numeric columns)
df_numeric = df.copy()
df_numeric['age'] = df_numeric['age'].fillna(df_numeric['age'].mean())
df_numeric['salary'] = df_numeric['salary'].fillna(df_numeric['salary'].mean())
print(f"\nFill numeric columns with mean:\n{df_numeric}")

# Fill with median
df_median = df.copy()
df_median['age'] = df_median['age'].fillna(df_median['age'].median())
print(f"\nFill 'age' with median:\n{df_median}")


# ----------------------------------------------------------------------------
# 1.4 Common Patterns for Missing Values
# ----------------------------------------------------------------------------
print("\n--- 1.4 Common Patterns for Missing Values ---")

df = pd.DataFrame({
    'age': [25, None, 35, 40, None, 30],
    'salary': [50000, 60000, None, 80000, 55000, None],
    'department': ['IT', 'Sales', 'IT', None, 'HR', 'Sales']
})
print(f"DataFrame:\n{df}")

# Pattern 1: Fill numeric with mean, categorical with mode
df_cleaned = df.copy()
df_cleaned['age'] = df_cleaned['age'].fillna(df_cleaned['age'].mean())
df_cleaned['salary'] = df_cleaned['salary'].fillna(df_cleaned['salary'].mean())
df_cleaned['department'] = df_cleaned['department'].fillna(df_cleaned['department'].mode()[0])
print(f"\nCleaned DataFrame:\n{df_cleaned}")

# Pattern 2: Drop rows with missing critical columns
df_dropped = df.dropna(subset=['age', 'salary'])
print(f"\nDrop rows missing age or salary:\n{df_dropped}")


# ============================================================================
# 2. AGGREGATIONS
# ============================================================================

print("\n" + "=" * 70)
print("2. AGGREGATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic Aggregation Functions
# ----------------------------------------------------------------------------
print("\n--- 2.1 Basic Aggregation Functions ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 28],
    'salary': [50000, 60000, 70000, 80000, 55000],
    'department': ['IT', 'Sales', 'IT', 'HR', 'Sales']
})
print(f"DataFrame:\n{df}")

# sum() - sum of values
print(f"\nsum():\n{df[['age', 'salary']].sum()}")

# mean() - mean of values
print(f"\nmean():\n{df[['age', 'salary']].mean()}")

# count() - count non-null values
print(f"\ncount():\n{df.count()}")

# min() and max()
print(f"\nmin():\n{df[['age', 'salary']].min()}")
print(f"\nmax():\n{df[['age', 'salary']].max()}")

# std() - standard deviation
print(f"\nstd():\n{df[['age', 'salary']].std()}")

# Multiple aggregations at once
print(f"\nMultiple aggregations:\n{df[['age', 'salary']].agg(['sum', 'mean', 'min', 'max'])}")


# ----------------------------------------------------------------------------
# 2.2 GroupBy Basics
# ----------------------------------------------------------------------------
print("\n--- 2.2 GroupBy Basics ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'department': ['IT', 'Sales', 'IT', 'HR', 'Sales', 'IT'],
    'salary': [50000, 60000, 70000, 80000, 55000, 65000],
    'age': [25, 30, 35, 40, 28, 32]
})
print(f"DataFrame:\n{df}")

# Basic groupby
grouped = df.groupby('department')
print(f"\ngroupby('department'):")
for name, group in grouped:
    print(f"\n{name}:\n{group}")

# Aggregation after groupby
print(f"\ngroupby('department')['salary'].mean():\n{df.groupby('department')['salary'].mean()}")

print(f"\ngroupby('department')['salary'].sum():\n{df.groupby('department')['salary'].sum()}")

print(f"\ngroupby('department')['salary'].count():\n{df.groupby('department')['salary'].count()}")


# ----------------------------------------------------------------------------
# 2.3 Multiple Aggregations with GroupBy
# ----------------------------------------------------------------------------
print("\n--- 2.3 Multiple Aggregations with GroupBy ---")

df = pd.DataFrame({
    'department': ['IT', 'Sales', 'IT', 'HR', 'Sales', 'IT'],
    'salary': [50000, 60000, 70000, 80000, 55000, 65000],
    'age': [25, 30, 35, 40, 28, 32]
})
print(f"DataFrame:\n{df}")

# Multiple aggregations on single column
print(f"\ngroupby('department')['salary'].agg(['sum', 'mean', 'count']):\n{df.groupby('department')['salary'].agg(['sum', 'mean', 'count'])}")

# Different aggregations on different columns
print(f"\ngroupby('department').agg({{'salary': 'mean', 'age': 'mean'}}):\n{df.groupby('department').agg({'salary': 'mean', 'age': 'mean'})}")

# Multiple aggregations on multiple columns
print(f"\ngroupby('department')[['salary', 'age']].agg(['mean', 'sum']):\n{df.groupby('department')[['salary', 'age']].agg(['mean', 'sum'])}")

# Named aggregations
print(f"\nNamed aggregations:\n{df.groupby('department').agg(avg_salary=('salary', 'mean'), total_salary=('salary', 'sum'))}")


# ----------------------------------------------------------------------------
# 2.4 GroupBy with Multiple Columns
# ----------------------------------------------------------------------------
print("\n--- 2.4 GroupBy with Multiple Columns ---")

df = pd.DataFrame({
    'department': ['IT', 'Sales', 'IT', 'HR', 'Sales', 'IT'],
    'location': ['NY', 'NY', 'CA', 'NY', 'CA', 'CA'],
    'salary': [50000, 60000, 70000, 80000, 55000, 65000]
})
print(f"DataFrame:\n{df}")

# Group by multiple columns
print(f"\ngroupby(['department', 'location'])['salary'].mean():\n{df.groupby(['department', 'location'])['salary'].mean()}")

# Reset index to convert to DataFrame
print(f"\nWith reset_index():\n{df.groupby(['department', 'location'])['salary'].mean().reset_index()}")


# ----------------------------------------------------------------------------
# 2.5 Common GroupBy Patterns
# ----------------------------------------------------------------------------
print("\n--- 2.5 Common GroupBy Patterns ---")

df = pd.DataFrame({
    'product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'quantity': [10, 20, 15, 25, 12, 18],
    'price': [100, 200, 110, 210, 105, 205]
})
df['revenue'] = df['quantity'] * df['price']
print(f"DataFrame:\n{df}")

# Pattern 1: Total revenue by product
print(f"\nTotal revenue by product:\n{df.groupby('product')['revenue'].sum()}")

# Pattern 2: Average quantity and price by product
print(f"\nAverage quantity and price by product:\n{df.groupby('product')[['quantity', 'price']].mean()}")

# Pattern 3: Count of records per product
print(f"\nCount per product:\n{df.groupby('product').size()}")

# Pattern 4: Multiple statistics
print(f"\nMultiple statistics:\n{df.groupby('product')['revenue'].agg(['sum', 'mean', 'count'])}")


# ============================================================================
# 3. MERGING - BASIC JOIN OPERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("3. MERGING - BASIC JOIN OPERATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Basic Merge
# ----------------------------------------------------------------------------
print("\n--- 3.1 Basic Merge ---")

# Create sample DataFrames
df1 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40]
})

df2 = pd.DataFrame({
    'id': [1, 2, 3, 5],
    'salary': [50000, 60000, 70000, 80000],
    'department': ['IT', 'Sales', 'IT', 'HR']
})

print(f"DataFrame 1:\n{df1}")
print(f"\nDataFrame 2:\n{df2}")

# Inner join (default)
print(f"\nInner join (default):\n{pd.merge(df1, df2, on='id')}")

# Left join
print(f"\nLeft join:\n{pd.merge(df1, df2, on='id', how='left')}")

# Right join
print(f"\nRight join:\n{pd.merge(df1, df2, on='id', how='right')}")

# Outer join (full outer)
print(f"\nOuter join:\n{pd.merge(df1, df2, on='id', how='outer')}")


# ----------------------------------------------------------------------------
# 3.2 Merge with Different Column Names
# ----------------------------------------------------------------------------
print("\n--- 3.2 Merge with Different Column Names ---")

df1 = pd.DataFrame({
    'employee_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

df2 = pd.DataFrame({
    'emp_id': [1, 2, 4],
    'salary': [50000, 60000, 70000]
})

print(f"DataFrame 1:\n{df1}")
print(f"\nDataFrame 2:\n{df2}")

# Merge with left_on and right_on
print(f"\nMerge with left_on and right_on:\n{pd.merge(df1, df2, left_on='employee_id', right_on='emp_id', how='left')}")


# ----------------------------------------------------------------------------
# 3.3 Merge on Multiple Columns
# ----------------------------------------------------------------------------
print("\n--- 3.3 Merge on Multiple Columns ---")

df1 = pd.DataFrame({
    'department': ['IT', 'Sales', 'IT'],
    'location': ['NY', 'CA', 'CA'],
    'employees': [10, 20, 15]
})

df2 = pd.DataFrame({
    'department': ['IT', 'Sales', 'IT'],
    'location': ['NY', 'CA', 'CA'],
    'budget': [100000, 200000, 150000]
})

print(f"DataFrame 1:\n{df1}")
print(f"\nDataFrame 2:\n{df2}")

# Merge on multiple columns
print(f"\nMerge on multiple columns:\n{pd.merge(df1, df2, on=['department', 'location'])}")


# ----------------------------------------------------------------------------
# 3.4 Merge with Suffixes
# ----------------------------------------------------------------------------
print("\n--- 3.4 Merge with Suffixes ---")

df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [10, 20, 30]
})

df2 = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [100, 200, 300]  # Same column name
})

print(f"DataFrame 1:\n{df1}")
print(f"\nDataFrame 2:\n{df2}")

# Merge with suffixes for overlapping columns
print(f"\nMerge with suffixes:\n{pd.merge(df1, df2, on='id', suffixes=('_left', '_right'))}")


# ----------------------------------------------------------------------------
# 3.5 Using DataFrame.merge() Method
# ----------------------------------------------------------------------------
print("\n--- 3.5 Using DataFrame.merge() Method ---")

df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

df2 = pd.DataFrame({
    'id': [1, 2, 4],
    'salary': [50000, 60000, 70000]
})

print(f"DataFrame 1:\n{df1}")
print(f"\nDataFrame 2:\n{df2}")

# Using merge() method
print(f"\ndf1.merge(df2, on='id', how='left'):\n{df1.merge(df2, on='id', how='left')}")


# ----------------------------------------------------------------------------
# 3.6 Common Merge Patterns
# ----------------------------------------------------------------------------
print("\n--- 3.6 Common Merge Patterns ---")

# Pattern 1: Enrich data with lookup table
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'dept_id': [1, 2, 1]
})

departments = pd.DataFrame({
    'dept_id': [1, 2, 3],
    'dept_name': ['IT', 'Sales', 'HR']
})

print(f"Employees:\n{employees}")
print(f"\nDepartments:\n{departments}")

# Enrich employees with department names
enriched = employees.merge(departments, on='dept_id', how='left')
print(f"\nEnriched employees:\n{enriched}")

# Pattern 2: Combine sales data
sales_q1 = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'q1_sales': [100, 200, 150]
})

sales_q2 = pd.DataFrame({
    'product': ['A', 'B', 'D'],
    'q2_sales': [120, 180, 160]
})

print(f"\nQ1 Sales:\n{sales_q1}")
print(f"\nQ2 Sales:\n{sales_q2}")

# Combine quarters
combined = sales_q1.merge(sales_q2, on='product', how='outer')
print(f"\nCombined sales:\n{combined}")


# ============================================================================
# 4. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("4. PRACTICAL EXAMPLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Complete Data Cleaning Workflow
# ----------------------------------------------------------------------------
print("\n--- 4.1 Complete Data Cleaning Workflow ---")

# Create messy data
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', None, 'David', 'Eve'],
    'age': [25, None, 35, 40, None],
    'salary': [50000, 60000, None, 80000, 55000],
    'department': ['IT', 'Sales', 'IT', None, 'HR']
})
print(f"Original messy data:\n{df}")

# Step 1: Check missing values
print(f"\nMissing values:\n{df.isna().sum()}")

# Step 2: Fill numeric columns with mean
df_cleaned = df.copy()
df_cleaned['age'] = df_cleaned['age'].fillna(df_cleaned['age'].mean())
df_cleaned['salary'] = df_cleaned['salary'].fillna(df_cleaned['salary'].mean())

# Step 3: Fill categorical columns
df_cleaned['name'] = df_cleaned['name'].fillna('Unknown')
df_cleaned['department'] = df_cleaned['department'].fillna('Unknown')

print(f"\nCleaned data:\n{df_cleaned}")


# ----------------------------------------------------------------------------
# 4.2 Aggregation Analysis
# ----------------------------------------------------------------------------
print("\n--- 4.2 Aggregation Analysis ---")

df = pd.DataFrame({
    'product': ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
    'category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics'],
    'quantity': [10, 20, 15, 25, 12, 18, 8],
    'price': [100, 50, 110, 55, 105, 52, 115]
})
df['revenue'] = df['quantity'] * df['price']
print(f"Sales data:\n{df}")

# Total revenue by product
print(f"\nTotal revenue by product:\n{df.groupby('product')['revenue'].sum()}")

# Average price by category
print(f"\nAverage price by category:\n{df.groupby('category')['price'].mean()}")

# Multiple aggregations by product
print(f"\nStatistics by product:\n{df.groupby('product').agg({'quantity': 'sum', 'revenue': ['sum', 'mean']})}")


# ----------------------------------------------------------------------------
# 4.3 Merging Multiple DataFrames
# ----------------------------------------------------------------------------
print("\n--- 4.3 Merging Multiple DataFrames ---")

# Employee data
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'dept_id': [1, 1, 2, 3]
})

# Department data
departments = pd.DataFrame({
    'dept_id': [1, 2, 3],
    'dept_name': ['IT', 'Sales', 'HR']
})

# Salary data
salaries = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'salary': [50000, 60000, 70000, 80000]
})

print(f"Employees:\n{employees}")
print(f"\nDepartments:\n{departments}")
print(f"\nSalaries:\n{salaries}")

# Merge step by step
step1 = employees.merge(departments, on='dept_id', how='left')
print(f"\nStep 1 - Add departments:\n{step1}")

final = step1.merge(salaries, on='emp_id', how='left')
print(f"\nFinal - Add salaries:\n{final}")


# ============================================================================
# 5. COMMON PATTERNS AND BEST PRACTICES
# ============================================================================

print("\n" + "=" * 70)
print("5. COMMON PATTERNS AND BEST PRACTICES")
print("=" * 70)

print("""
1. MISSING VALUES:
   - Check first: df.isna().sum()
   - Drop: df.dropna() or df.dropna(subset=['col'])
   - Fill: df.fillna(value) or df.fillna({'col': value})
   - Fill numeric: df.fillna(df.mean())
   - Fill categorical: df.fillna(df.mode()[0])

2. AGGREGATIONS:
   - Basic: df.sum(), df.mean(), df.count()
   - GroupBy: df.groupby('col').agg({'col2': 'mean'})
   - Multiple: df.groupby('col')['col2'].agg(['sum', 'mean'])
   - Named: df.groupby('col').agg(avg=('col2', 'mean'))

3. MERGING:
   - Inner: how='inner' (default)
   - Left: how='left'
   - Right: how='right'
   - Outer: how='outer'
   - Different columns: left_on, right_on
   - Multiple keys: on=['col1', 'col2']

4. BEST PRACTICES:
   - Always check missing values first
   - Use appropriate fill strategy
   - GroupBy before aggregating
   - Specify how='left' or 'inner' explicitly
   - Use suffixes when columns overlap
   - Check merge results with shape
""")


# ============================================================================
# 6. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Handle missing values
print("\n--- Exercise 1: Handle Missing Values ---")
df = pd.DataFrame({
    'age': [25, None, 35, None, 30],
    'salary': [50000, 60000, None, 80000, None]
})
print(f"Original:\n{df}")
df['age'] = df['age'].fillna(df['age'].mean())
df['salary'] = df['salary'].fillna(df['salary'].mean())
print(f"After filling:\n{df}")

# Exercise 2: GroupBy aggregation
print("\n--- Exercise 2: GroupBy Aggregation ---")
df = pd.DataFrame({
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250]
})
result = df.groupby('product')['sales'].sum()
print(f"Total sales by product:\n{result}")

# Exercise 3: Merge DataFrames
print("\n--- Exercise 3: Merge DataFrames ---")
df1 = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'id': [1, 3], 'salary': [50000, 60000]})
result = df1.merge(df2, on='id', how='left')
print(f"Merged:\n{result}")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. MISSING VALUES:
   - isna() / isnull() to detect
   - dropna() to remove
   - fillna() to fill
   - Common: fill numeric with mean, categorical with mode

2. AGGREGATIONS:
   - sum(), mean(), count(), min(), max()
   - groupby() for grouped aggregations
   - agg() for multiple aggregations
   - Can aggregate on single or multiple columns

3. GROUPBY:
   - df.groupby('col')['col2'].mean()
   - df.groupby('col').agg({'col2': 'mean'})
   - Multiple groups: groupby(['col1', 'col2'])
   - reset_index() to convert to DataFrame

4. MERGING:
   - pd.merge(df1, df2, on='key')
   - how: 'inner', 'left', 'right', 'outer'
   - left_on, right_on for different column names
   - suffixes for overlapping columns

5. COMMON PATTERNS:
   - Clean data → Group → Aggregate → Merge
   - Fill missing before aggregating
   - Use left join to preserve all rows
   - Check results with shape and head()

6. INTERVIEW TIPS:
   - Know difference between join types
   - Understand when to use groupby
   - Know how to handle missing values
   - Practice combining operations
   - Check edge cases (empty groups, missing keys)
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Common Operations Guide Ready!")
    print("=" * 70)
