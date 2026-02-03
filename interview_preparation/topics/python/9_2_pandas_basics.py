"""
Pandas Basics - Interview Preparation
Topic 9.2: Pandas Basics

This module covers:
- Series: Creating, indexing, basic operations
- DataFrame: Creating, indexing, selecting columns/rows
- Basic Operations: head(), tail(), describe(), info()
- Filtering: Boolean indexing, query()
"""

import pandas as pd
import numpy as np

# ============================================================================
# 1. SERIES
# ============================================================================

print("=" * 70)
print("1. SERIES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Creating Series
# ----------------------------------------------------------------------------
print("\n--- 1.1 Creating Series ---")

# From list
series1 = pd.Series([1, 2, 3, 4, 5])
print(f"Series from list:\n{series1}")

# With custom index
series2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(f"\nSeries with custom index:\n{series2}")

# From dictionary
series3 = pd.Series({'Alice': 25, 'Bob': 30, 'Charlie': 35})
print(f"\nSeries from dictionary:\n{series3}")

# With name
series4 = pd.Series([1, 2, 3, 4], name='numbers')
print(f"\nSeries with name:\n{series4}")
print(f"Name: {series4.name}")


# ----------------------------------------------------------------------------
# 1.2 Series Indexing
# ----------------------------------------------------------------------------
print("\n--- 1.2 Series Indexing ---")

series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(f"Series:\n{series}")

# Access by label
print(f"\nseries['a']: {series['a']}")
print(f"series['c']: {series['c']}")

# Access by position
print(f"series[0]: {series[0]}")
print(f"series[-1]: {series[-1]}")

# Slicing
print(f"\nseries['a':'c']:\n{series['a':'c']}")
print(f"series[0:3]:\n{series[0:3]}")

# Multiple labels
print(f"\nseries[['a', 'c', 'e']]:\n{series[['a', 'c', 'e']]}")


# ----------------------------------------------------------------------------
# 1.3 Series Basic Operations
# ----------------------------------------------------------------------------
print("\n--- 1.3 Series Basic Operations ---")

series = pd.Series([1, 2, 3, 4, 5])
print(f"Series: {series}")

# Arithmetic operations
print(f"\nseries + 10:\n{series + 10}")
print(f"series * 2:\n{series * 2}")
print(f"series ** 2:\n{series ** 2}")

# Operations with another series
series2 = pd.Series([10, 20, 30, 40, 50])
print(f"\nseries + series2:\n{series + series2}")

# Statistical operations
print(f"\nMean: {series.mean()}")
print(f"Sum: {series.sum()}")
print(f"Min: {series.min()}")
print(f"Max: {series.max()}")
print(f"Std: {series.std()}")

# Boolean operations
print(f"\nseries > 3:\n{series > 3}")
print(f"series[series > 3]:\n{series[series > 3]}")


# ----------------------------------------------------------------------------
# 1.4 Series Attributes
# ----------------------------------------------------------------------------
print("\n--- 1.4 Series Attributes ---")

series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'], name='numbers')
print(f"Series:\n{series}")

print(f"\nAttributes:")
print(f"  index: {series.index}")
print(f"  values: {series.values}")
print(f"  name: {series.name}")
print(f"  dtype: {series.dtype}")
print(f"  size: {series.size}")
print(f"  shape: {series.shape}")


# ============================================================================
# 2. DATAFRAME
# ============================================================================

print("\n" + "=" * 70)
print("2. DATAFRAME")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Creating DataFrame
# ----------------------------------------------------------------------------
print("\n--- 2.1 Creating DataFrame ---")

# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'city': ['New York', 'London', 'Tokyo', 'Paris']
}
df1 = pd.DataFrame(data)
print(f"DataFrame from dictionary:\n{df1}")

# From list of lists
data_list = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df2 = pd.DataFrame(data_list, columns=['name', 'age'])
print(f"\nDataFrame from list:\n{df2}")

# From list of dictionaries
data_dict_list = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30},
    {'name': 'Charlie', 'age': 35}
]
df3 = pd.DataFrame(data_dict_list)
print(f"\nDataFrame from list of dicts:\n{df3}")

# From NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df4 = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print(f"\nDataFrame from NumPy array:\n{df4}")


# ----------------------------------------------------------------------------
# 2.2 DataFrame Indexing - Selecting Columns
# ----------------------------------------------------------------------------
print("\n--- 2.2 DataFrame Indexing - Selecting Columns ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'city': ['New York', 'London', 'Tokyo', 'Paris'],
    'salary': [50000, 60000, 70000, 80000]
})
print(f"DataFrame:\n{df}")

# Select single column (returns Series)
print(f"\ndf['name']:\n{df['name']}")
print(f"Type: {type(df['name'])}")

# Select multiple columns (returns DataFrame)
print(f"\ndf[['name', 'age']]:\n{df[['name', 'age']]}")

# Using dot notation (only for valid Python identifiers)
print(f"\ndf.name:\n{df.name}")

# Add new column
df['bonus'] = df['salary'] * 0.1
print(f"\nAfter adding 'bonus' column:\n{df}")


# ----------------------------------------------------------------------------
# 2.3 DataFrame Indexing - Selecting Rows
# ----------------------------------------------------------------------------
print("\n--- 2.3 DataFrame Indexing - Selecting Rows ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 28],
    'city': ['New York', 'London', 'Tokyo', 'Paris', 'Berlin'],
    'salary': [50000, 60000, 70000, 80000, 55000]
})
print(f"DataFrame:\n{df}")

# Using iloc (integer position-based)
print(f"\ndf.iloc[0]:\n{df.iloc[0]}")
print(f"df.iloc[0:2]:\n{df.iloc[0:2]}")
print(f"df.iloc[0, 1]: {df.iloc[0, 1]}")  # Row 0, Column 1
print(f"df.iloc[0:2, 0:2]:\n{df.iloc[0:2, 0:2]}")

# Using loc (label-based)
print(f"\ndf.loc[0]:\n{df.loc[0]}")
print(f"df.loc[0:2]:\n{df.loc[0:2]}")
print(f"df.loc[0, 'name']: {df.loc[0, 'name']}")
print(f"df.loc[0:2, ['name', 'age']]:\n{df.loc[0:2, ['name', 'age']]}")


# ----------------------------------------------------------------------------
# 2.4 DataFrame Attributes
# ----------------------------------------------------------------------------
print("\n--- 2.4 DataFrame Attributes ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'London', 'Tokyo']
})
print(f"DataFrame:\n{df}")

print(f"\nAttributes:")
print(f"  shape: {df.shape}")
print(f"  size: {df.size}")
print(f"  columns: {df.columns.tolist()}")
print(f"  index: {df.index.tolist()}")
print(f"  dtypes:\n{df.dtypes}")
print(f"  values:\n{df.values}")


# ============================================================================
# 3. BASIC OPERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("3. BASIC OPERATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 head() and tail()
# ----------------------------------------------------------------------------
print("\n--- 3.1 head() and tail() ---")

df = pd.DataFrame({
    'name': [f'Person_{i}' for i in range(10)],
    'age': np.random.randint(20, 50, 10),
    'salary': np.random.randint(40000, 100000, 10)
})
print(f"DataFrame shape: {df.shape}")

# head() - first n rows (default 5)
print(f"\ndf.head():\n{df.head()}")
print(f"df.head(3):\n{df.head(3)}")

# tail() - last n rows (default 5)
print(f"\ndf.tail():\n{df.tail()}")
print(f"df.tail(3):\n{df.tail(3)}")


# ----------------------------------------------------------------------------
# 3.2 describe()
# ----------------------------------------------------------------------------
print("\n--- 3.2 describe() ---")

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 28, 32, 45],
    'salary': [50000, 60000, 70000, 80000, 55000, 65000, 90000],
    'experience': [2, 5, 8, 12, 3, 6, 15]
})
print(f"DataFrame:\n{df}")

# describe() - statistical summary
print(f"\ndf.describe():\n{df.describe()}")

# describe() for specific columns
print(f"\ndf[['age', 'salary']].describe():\n{df[['age', 'salary']].describe()}")

# describe() with include='all' for all columns
print(f"\ndf.describe(include='all'):\n{df.describe(include='all')}")


# ----------------------------------------------------------------------------
# 3.3 info()
# ----------------------------------------------------------------------------
print("\n--- 3.3 info() ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000],
    'city': ['New York', 'London', 'Tokyo']
})
print(f"DataFrame:\n{df}")

print("\ndf.info():")
df.info()


# ----------------------------------------------------------------------------
# 3.4 Other Useful Methods
# ----------------------------------------------------------------------------
print("\n--- 3.4 Other Useful Methods ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'salary': [50000, 60000, 70000, 80000]
})
print(f"DataFrame:\n{df}")

print(f"\nOther methods:")
print(f"  df.shape: {df.shape}")
print(f"  df.size: {df.size}")
print(f"  df.empty: {df.empty}")
print(f"  df.ndim: {df.ndim}")
print(f"  df.columns.tolist(): {df.columns.tolist()}")
print(f"  df.index.tolist(): {df.index.tolist()}")

# Value counts for Series
print(f"\ndf['age'].value_counts():\n{df['age'].value_counts()}")

# Unique values
print(f"\ndf['age'].unique(): {df['age'].unique()}")
print(f"df['age'].nunique(): {df['age'].nunique()}")


# ============================================================================
# 4. FILTERING
# ============================================================================

print("\n" + "=" * 70)
print("4. FILTERING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Boolean Indexing
# ----------------------------------------------------------------------------
print("\n--- 4.1 Boolean Indexing ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 28],
    'city': ['New York', 'London', 'Tokyo', 'Paris', 'Berlin'],
    'salary': [50000, 60000, 70000, 80000, 55000]
})
print(f"DataFrame:\n{df}")

# Single condition
print(f"\nFilter: age > 30")
filtered = df[df['age'] > 30]
print(filtered)

# Multiple conditions (use &, |, ~)
print(f"\nFilter: age > 30 AND salary > 60000")
filtered = df[(df['age'] > 30) & (df['salary'] > 60000)]
print(filtered)

# Using OR
print(f"\nFilter: age < 30 OR salary > 70000")
filtered = df[(df['age'] < 30) | (df['salary'] > 70000)]
print(filtered)

# Using NOT
print(f"\nFilter: NOT (age > 30)")
filtered = df[~(df['age'] > 30)]
print(filtered)

# String conditions
print(f"\nFilter: city == 'London'")
filtered = df[df['city'] == 'London']
print(filtered)

# Using isin()
print(f"\nFilter: city in ['London', 'Paris']")
filtered = df[df['city'].isin(['London', 'Paris'])]
print(filtered)

# Using contains() for string matching
print(f"\nFilter: name contains 'a'")
filtered = df[df['name'].str.contains('a', case=False)]
print(filtered)


# ----------------------------------------------------------------------------
# 4.2 query() Method
# ----------------------------------------------------------------------------
print("\n--- 4.2 query() Method ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 28],
    'salary': [50000, 60000, 70000, 80000, 55000]
})
print(f"DataFrame:\n{df}")

# Simple query
print(f"\nquery('age > 30'):")
result = df.query('age > 30')
print(result)

# Multiple conditions
print(f"\nquery('age > 30 and salary > 60000'):")
result = df.query('age > 30 and salary > 60000')
print(result)

# Using OR
print(f"\nquery('age < 30 or salary > 70000'):")
result = df.query('age < 30 or salary > 70000')
print(result)

# Using variables
threshold = 30
print(f"\nquery('age > @threshold'):")
result = df.query('age > @threshold')
print(result)


# ----------------------------------------------------------------------------
# 4.3 Filtering Examples
# ----------------------------------------------------------------------------
print("\n--- 4.3 Filtering Examples ---")

df = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D', 'E'],
    'price': [10, 20, 30, 40, 50],
    'quantity': [100, 200, 150, 300, 250],
    'category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics']
})
print(f"DataFrame:\n{df}")

# Filter by multiple conditions
print(f"\nProducts with price > 20 and quantity > 150:")
filtered = df[(df['price'] > 20) & (df['quantity'] > 150)]
print(filtered)

# Filter by category
print(f"\nElectronics products:")
filtered = df[df['category'] == 'Electronics']
print(filtered)

# Filter and select specific columns
print(f"\nProduct names where price > 25:")
filtered = df[df['price'] > 25]['product']
print(filtered)

# Complex filtering
print(f"\nProducts in Electronics category OR price < 25:")
filtered = df[(df['category'] == 'Electronics') | (df['price'] < 25)]
print(filtered)


# ============================================================================
# 5. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICAL EXAMPLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Data Exploration Workflow
# ----------------------------------------------------------------------------
print("\n--- 5.1 Data Exploration Workflow ---")

# Create sample data
df = pd.DataFrame({
    'id': range(1, 11),
    'name': [f'Employee_{i}' for i in range(1, 11)],
    'age': np.random.randint(22, 55, 10),
    'department': np.random.choice(['Sales', 'IT', 'HR'], 10),
    'salary': np.random.randint(40000, 100000, 10),
    'years_experience': np.random.randint(1, 20, 10)
})
print(f"DataFrame:\n{df}")

# Basic exploration
print(f"\n1. Shape: {df.shape}")
print(f"2. Columns: {df.columns.tolist()}")
print(f"\n3. First 3 rows:\n{df.head(3)}")
print(f"\n4. Last 3 rows:\n{df.tail(3)}")
print(f"\n5. Summary statistics:\n{df.describe()}")
print(f"\n6. Data types:\n{df.dtypes}")

# Filtering
print(f"\n7. Employees in IT department:")
it_employees = df[df['department'] == 'IT']
print(it_employees)

print(f"\n8. Employees with salary > 60000:")
high_salary = df[df['salary'] > 60000]
print(high_salary)


# ----------------------------------------------------------------------------
# 5.2 Working with Missing Data
# ----------------------------------------------------------------------------
print("\n--- 5.2 Working with Missing Data ---")

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, None, 35, 40, 28],
    'salary': [50000, 60000, None, 80000, 55000],
    'city': ['New York', 'London', None, 'Paris', 'Berlin']
})
print(f"DataFrame with missing values:\n{df}")

print(f"\nisna():\n{df.isna()}")
print(f"\nisnull():\n{df.isnull()}")
print(f"\nnotna():\n{df.notna()}")

# Count missing values
print(f"\nMissing values per column:\n{df.isna().sum()}")

# Filter rows with no missing values
print(f"\nRows with no missing values:\n{df.dropna()}")

# Filter rows with missing values in specific column
print(f"\nRows with missing age:\n{df[df['age'].isna()]}")


# ============================================================================
# 6. COMMON PATTERNS AND BEST PRACTICES
# ============================================================================

print("\n" + "=" * 70)
print("6. COMMON PATTERNS AND BEST PRACTICES")
print("=" * 70)

print("""
1. SERIES:
   - Use for single column of data
   - Indexed by labels or integers
   - Supports vectorized operations
   - Access with [] or .loc[]

2. DATAFRAME:
   - Use for tabular data (multiple columns)
   - Select columns: df['col'] or df[['col1', 'col2']]
   - Select rows: df.loc[] (label) or df.iloc[] (position)
   - Combine: df.loc[rows, cols]

3. INDEXING:
   - Use [] for column selection
   - Use .loc[] for label-based row/column selection
   - Use .iloc[] for position-based selection
   - Boolean indexing: df[df['col'] > value]

4. FILTERING:
   - Boolean indexing: df[condition]
   - Multiple conditions: use & (and), | (or), ~ (not)
   - query() method: df.query('condition')
   - isin() for multiple values: df[df['col'].isin([val1, val2])]

5. BASIC OPERATIONS:
   - head(n): first n rows
   - tail(n): last n rows
   - describe(): statistical summary
   - info(): data types and memory usage
   - shape: (rows, columns)
   - columns: column names

6. BEST PRACTICES:
   - Always check shape and info() first
   - Use describe() for numeric columns
   - Use boolean indexing instead of loops
   - Use query() for complex conditions
   - Be careful with chained indexing
""")


# ============================================================================
# 7. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Create and filter Series
print("\n--- Exercise 1: Create and Filter Series ---")
series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(f"Series:\n{series}")
print(f"Values > 30:\n{series[series > 30]}")

# Exercise 2: Create DataFrame and select columns
print("\n--- Exercise 2: Create DataFrame and Select Columns ---")
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})
print(f"DataFrame:\n{df}")
print(f"Name and age columns:\n{df[['name', 'age']]}")

# Exercise 3: Filter DataFrame
print("\n--- Exercise 3: Filter DataFrame ---")
df = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D'],
    'price': [10, 20, 30, 40],
    'stock': [100, 200, 50, 300]
})
print(f"DataFrame:\n{df}")
print(f"Products with price > 20:\n{df[df['price'] > 20]}")

# Exercise 4: Use describe() and info()
print("\n--- Exercise 4: Use describe() and info() ---")
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 28],
    'salary': [50000, 60000, 70000, 80000, 55000]
})
print(f"describe():\n{df.describe()}")
print(f"\ninfo():")
df.info()


# ============================================================================
# 8. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("8. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. SERIES:
   - One-dimensional labeled array
   - Created from list, dict, or array
   - Indexed by labels or position
   - Supports vectorized operations

2. DATAFRAME:
   - Two-dimensional labeled structure
   - Created from dict, list of lists, or arrays
   - Columns are Series
   - Rows and columns are indexed

3. INDEXING:
   - Columns: df['col'] or df[['col1', 'col2']]
   - Rows: df.loc[] (label) or df.iloc[] (position)
   - Combined: df.loc[rows, cols]
   - Boolean: df[df['col'] > value]

4. BASIC OPERATIONS:
   - head(n): first n rows
   - tail(n): last n rows
   - describe(): summary statistics
   - info(): data types and info
   - shape: (rows, columns)

5. FILTERING:
   - Boolean indexing: df[condition]
   - Multiple conditions: & (and), | (or), ~ (not)
   - query(): df.query('condition')
   - isin(): df[df['col'].isin([val1, val2])]
   - String methods: df[df['col'].str.contains('text')]

6. COMMON PATTERNS:
   - df[df['col'] > threshold]
   - df[(df['col1'] > val1) & (df['col2'] < val2)]
   - df.query('col1 > val1 and col2 < val2')
   - df[df['col'].isin([val1, val2])]

7. INTERVIEW TIPS:
   - Know difference between loc and iloc
   - Understand boolean indexing syntax
   - Know when to use query() vs boolean indexing
   - Be familiar with describe() and info()
   - Understand Series vs DataFrame
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Pandas Basics Guide Ready!")
    print("=" * 70)
