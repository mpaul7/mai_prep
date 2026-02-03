"""
NumPy Basics - Interview Preparation
Topic 9.1: NumPy Basics

This module covers:
- Arrays: Creating arrays, array operations
- Array Indexing: Slicing, boolean indexing
- Array Methods: shape, dtype, reshape
- Basic Operations: Element-wise operations, broadcasting
"""

import numpy as np

# ============================================================================
# 1. CREATING ARRAYS
# ============================================================================

print("=" * 70)
print("1. CREATING ARRAYS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 From Python Lists
# ----------------------------------------------------------------------------
print("\n--- 1.1 From Python Lists ---")

# 1D array
arr1d = np.array([1, 2, 3, 4, 5])
print(f"1D array: {arr1d}")
print(f"Type: {type(arr1d)}")
print(f"Shape: {arr1d.shape}")

# 2D array (matrix)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D array:\n{arr2d}")
print(f"Shape: {arr2d.shape}")

# 3D array
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n3D array:\n{arr3d}")
print(f"Shape: {arr3d.shape}")


# ----------------------------------------------------------------------------
# 1.2 Using NumPy Functions
# ----------------------------------------------------------------------------
print("\n--- 1.2 Using NumPy Functions ---")

# zeros - array filled with zeros
zeros_1d = np.zeros(5)
zeros_2d = np.zeros((3, 4))
print(f"zeros(5): {zeros_1d}")
print(f"zeros((3, 4)):\n{zeros_2d}")

# ones - array filled with ones
ones_1d = np.ones(5)
ones_2d = np.ones((2, 3))
print(f"\nones(5): {ones_1d}")
print(f"ones((2, 3)):\n{ones_2d}")

# full - array filled with specific value
full_arr = np.full((2, 3), 7)
print(f"\nfull((2, 3), 7):\n{full_arr}")

# arange - similar to range()
arr_range = np.arange(0, 10, 2)
print(f"\narange(0, 10, 2): {arr_range}")

# linspace - evenly spaced numbers
arr_linspace = np.linspace(0, 1, 5)
print(f"linspace(0, 1, 5): {arr_linspace}")

# random - random numbers
random_arr = np.random.random((2, 3))
print(f"\nrandom((2, 3)):\n{random_arr}")

# random integers
random_int = np.random.randint(0, 10, (2, 3))
print(f"randint(0, 10, (2, 3)):\n{random_int}")


# ----------------------------------------------------------------------------
# 1.3 Array Properties
# ----------------------------------------------------------------------------
print("\n--- 1.3 Array Properties ---")

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")          # Dimensions
print(f"Size: {arr.size}")             # Total elements
print(f"ndim: {arr.ndim}")             # Number of dimensions
print(f"dtype: {arr.dtype}")           # Data type
print(f"itemsize: {arr.itemsize}")     # Bytes per element
print(f"nbytes: {arr.nbytes}")         # Total bytes


# ----------------------------------------------------------------------------
# 1.4 Data Types (dtype)
# ----------------------------------------------------------------------------
print("\n--- 1.4 Data Types (dtype) ---")

# Specify dtype when creating
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1, 2, 3], dtype=np.float64)
arr_bool = np.array([True, False, True], dtype=bool)

print(f"int32 array: {arr_int}, dtype: {arr_int.dtype}")
print(f"float64 array: {arr_float}, dtype: {arr_float.dtype}")
print(f"bool array: {arr_bool}, dtype: {arr_bool.dtype}")

# Convert dtype
arr_converted = arr_int.astype(np.float64)
print(f"\nConverted to float64: {arr_converted}, dtype: {arr_converted.dtype}")


# ============================================================================
# 2. ARRAY INDEXING
# ============================================================================

print("\n" + "=" * 70)
print("2. ARRAY INDEXING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic Indexing
# ----------------------------------------------------------------------------
print("\n--- 2.1 Basic Indexing ---")

arr = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr}")

# Single element
print(f"arr[0] = {arr[0]}")
print(f"arr[-1] = {arr[-1]}")

# 2D array indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D Array:\n{arr2d}")
print(f"arr2d[0, 0] = {arr2d[0, 0]}")
print(f"arr2d[1, 2] = {arr2d[1, 2]}")
print(f"arr2d[0] = {arr2d[0]}")  # First row
print(f"arr2d[:, 0] = {arr2d[:, 0]}")  # First column


# ----------------------------------------------------------------------------
# 2.2 Slicing
# ----------------------------------------------------------------------------
print("\n--- 2.2 Slicing ---")

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"Array: {arr}")

# Basic slicing
print(f"arr[2:5] = {arr[2:5]}")
print(f"arr[:5] = {arr[:5]}")
print(f"arr[5:] = {arr[5:]}")
print(f"arr[::2] = {arr[::2]}")  # Every 2nd element
print(f"arr[::-1] = {arr[::-1]}")  # Reverse

# 2D slicing
arr2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"\n2D Array:\n{arr2d}")
print(f"arr2d[0:2, 1:3]:\n{arr2d[0:2, 1:3]}")  # Submatrix
print(f"arr2d[:, 1:3]:\n{arr2d[:, 1:3]}")  # All rows, columns 1-2
print(f"arr2d[1:, :]:\n{arr2d[1:, :]}")  # Rows 1 onwards


# ----------------------------------------------------------------------------
# 2.3 Boolean Indexing
# ----------------------------------------------------------------------------
print("\n--- 2.3 Boolean Indexing ---")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")

# Create boolean mask
mask = arr > 5
print(f"Mask (arr > 5): {mask}")

# Use mask to filter
filtered = arr[mask]
print(f"Filtered (arr[mask]): {filtered}")

# Direct boolean indexing
print(f"arr[arr > 5]: {arr[arr > 5]}")
print(f"arr[arr % 2 == 0]: {arr[arr % 2 == 0]}")  # Even numbers
print(f"arr[(arr > 3) & (arr < 8)]: {arr[(arr > 3) & (arr < 8)]}")  # Multiple conditions

# Modify elements using boolean indexing
arr_copy = arr.copy()
arr_copy[arr_copy > 5] = 0
print(f"\nAfter setting > 5 to 0: {arr_copy}")


# ----------------------------------------------------------------------------
# 2.4 Fancy Indexing (Integer Array Indexing)
# ----------------------------------------------------------------------------
print("\n--- 2.4 Fancy Indexing (Integer Array Indexing) ---")

arr = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr}")

# Select specific indices
indices = [0, 2, 4]
print(f"arr[[0, 2, 4]]: {arr[[0, 2, 4]]}")

# 2D fancy indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D Array:\n{arr2d}")
print(f"arr2d[[0, 2], [0, 2]]: {arr2d[[0, 2], [0, 2]]}")  # (0,0) and (2,2)
print(f"arr2d[[0, 2]]:\n{arr2d[[0, 2]]}")  # Rows 0 and 2


# ============================================================================
# 3. ARRAY METHODS
# ============================================================================

print("\n" + "=" * 70)
print("3. ARRAY METHODS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Shape Methods
# ----------------------------------------------------------------------------
print("\n--- 3.1 Shape Methods ---")

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array:\n{arr}")
print(f"shape: {arr.shape}")
print(f"ndim: {arr.ndim}")
print(f"size: {arr.size}")

# Reshape
reshaped = arr.reshape(3, 2)
print(f"\nReshaped to (3, 2):\n{reshaped}")

# Flatten
flattened = arr.flatten()
print(f"Flattened: {flattened}")

# Reshape with -1 (auto-calculate)
reshaped_auto = arr.reshape(-1)  # 1D
print(f"Reshape(-1): {reshaped_auto}")

reshaped_col = arr.reshape(-1, 1)  # Column vector
print(f"Reshape(-1, 1):\n{reshaped_col}")


# ----------------------------------------------------------------------------
# 3.2 Data Type Methods
# ----------------------------------------------------------------------------
print("\n--- 3.2 Data Type Methods ---")

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"dtype: {arr.dtype}")

# Convert to different types
arr_float = arr.astype(np.float64)
print(f"astype(float64): {arr_float}, dtype: {arr_float.dtype}")

arr_str = arr.astype(str)
print(f"astype(str): {arr_str}, dtype: {arr_str.dtype}")


# ----------------------------------------------------------------------------
# 3.3 Array Manipulation Methods
# ----------------------------------------------------------------------------
print("\n--- 3.3 Array Manipulation Methods ---")

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate
concatenated = np.concatenate([arr1, arr2])
print(f"concatenate([{arr1}, {arr2}]): {concatenated}")

# Stack (along new axis)
stacked_v = np.vstack([arr1, arr2])  # Vertical stack
print(f"\nvstack:\n{stacked_v}")

stacked_h = np.hstack([arr1, arr2])  # Horizontal stack
print(f"hstack: {stacked_h}")

# Split
arr = np.array([1, 2, 3, 4, 5, 6])
split_arr = np.split(arr, 3)
print(f"\nsplit into 3: {split_arr}")


# ----------------------------------------------------------------------------
# 3.4 Statistical Methods
# ----------------------------------------------------------------------------
print("\n--- 3.4 Statistical Methods ---")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")

print(f"mean(): {arr.mean()}")
print(f"sum(): {arr.sum()}")
print(f"min(): {arr.min()}")
print(f"max(): {arr.max()}")
print(f"std(): {arr.std()}")
print(f"var(): {arr.var()}")

# 2D array statistics
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D Array:\n{arr2d}")
print(f"mean(axis=0): {arr2d.mean(axis=0)}")  # Mean along columns
print(f"mean(axis=1): {arr2d.mean(axis=1)}")  # Mean along rows
print(f"mean(): {arr2d.mean()}")  # Overall mean


# ============================================================================
# 4. BASIC OPERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("4. BASIC OPERATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Element-wise Operations
# ----------------------------------------------------------------------------
print("\n--- 4.1 Element-wise Operations ---")

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# Arithmetic operations
print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
print(f"arr1 + arr2: {arr1 + arr2}")
print(f"arr1 - arr2: {arr1 - arr2}")
print(f"arr1 * arr2: {arr1 * arr2}")
print(f"arr1 / arr2: {arr1 / arr2}")
print(f"arr1 ** 2: {arr1 ** 2}")

# Comparison operations
print(f"\narr1 > 2: {arr1 > 2}")
print(f"arr1 == arr2: {arr1 == arr2}")
print(f"arr1 != arr2: {arr1 != arr2}")


# ----------------------------------------------------------------------------
# 4.2 Scalar Operations
# ----------------------------------------------------------------------------
print("\n--- 4.2 Scalar Operations ---")

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"arr + 10: {arr + 10}")
print(f"arr * 2: {arr * 2}")
print(f"arr ** 2: {arr ** 2}")
print(f"arr > 3: {arr > 3}")


# ----------------------------------------------------------------------------
# 4.3 Universal Functions (ufunc)
# ----------------------------------------------------------------------------
print("\n--- 4.3 Universal Functions (ufunc) ---")

arr = np.array([1, 4, 9, 16, 25])
print(f"Array: {arr}")

print(f"np.sqrt(arr): {np.sqrt(arr)}")
print(f"np.sin(arr): {np.sin(arr)}")
print(f"np.exp(arr): {np.exp(arr)}")
print(f"np.log(arr): {np.log(arr)}")
print(f"np.abs([-1, -2, 3]): {np.abs(np.array([-1, -2, 3]))}")


# ----------------------------------------------------------------------------
# 4.4 Broadcasting
# ----------------------------------------------------------------------------
print("\n--- 4.4 Broadcasting ---")
print("""
Broadcasting allows NumPy to perform operations on arrays of different shapes.
Rules:
1. Arrays are aligned from the right
2. Dimensions must match or be 1
3. Missing dimensions are treated as 1
""")

# Example 1: Array + scalar
arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
print(f"Array:\n{arr}")
print(f"Scalar: {scalar}")
print(f"Array + scalar:\n{arr + scalar}")

# Example 2: Array + 1D array
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr1d = np.array([10, 20, 30])
print(f"\n2D Array:\n{arr2d}")
print(f"1D Array: {arr1d}")
print(f"2D + 1D (broadcast along rows):\n{arr2d + arr1d}")

# Example 3: Column vector + row vector
col = np.array([[1], [2], [3]])
row = np.array([10, 20, 30])
print(f"\nColumn:\n{col}")
print(f"Row: {row}")
print(f"Column + Row:\n{col + row}")

# Example 4: Broadcasting failure
print("\nBroadcasting failure example:")
try:
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([1, 2, 3])
    result = arr1 + arr2
except ValueError as e:
    print(f"  Error: {e}")
    print("  Reason: Shapes (2, 2) and (3,) cannot be broadcast")


# ----------------------------------------------------------------------------
# 4.5 Matrix Operations
# ----------------------------------------------------------------------------
print("\n--- 4.5 Matrix Operations ---")

# Element-wise multiplication (not matrix multiplication)
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print(f"Array 1:\n{arr1}")
print(f"Array 2:\n{arr2}")
print(f"Element-wise multiplication (*):\n{arr1 * arr2}")

# Matrix multiplication
matrix_mult = np.dot(arr1, arr2)
print(f"\nMatrix multiplication (dot):\n{matrix_mult}")

# Using @ operator (Python 3.5+)
matrix_mult_at = arr1 @ arr2
print(f"Matrix multiplication (@):\n{matrix_mult_at}")

# Transpose
print(f"\nTranspose:\n{arr1.T}")

# Dot product of vectors
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
dot_product = np.dot(vec1, vec2)
print(f"\nDot product: {dot_product}")


# ============================================================================
# 5. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICAL EXAMPLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Filtering and Conditional Operations
# ----------------------------------------------------------------------------
print("\n--- 5.1 Filtering and Conditional Operations ---")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Find elements greater than 5
greater_than_5 = arr[arr > 5]
print(f"Elements > 5: {greater_than_5}")

# Replace values conditionally
arr_copy = arr.copy()
arr_copy[arr_copy > 5] = 0
print(f"Replace > 5 with 0: {arr_copy}")

# Using np.where
result = np.where(arr > 5, arr, 0)
print(f"np.where(arr > 5, arr, 0): {result}")


# ----------------------------------------------------------------------------
# 5.2 Array Aggregation
# ----------------------------------------------------------------------------
print("\n--- 5.2 Array Aggregation ---")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Array:\n{arr}")

print(f"Sum of all elements: {arr.sum()}")
print(f"Sum along axis=0 (columns): {arr.sum(axis=0)}")
print(f"Sum along axis=1 (rows): {arr.sum(axis=1)}")
print(f"Mean: {arr.mean()}")
print(f"Max: {arr.max()}")
print(f"Min: {arr.min()}")


# ----------------------------------------------------------------------------
# 5.3 Array Reshaping for ML
# ----------------------------------------------------------------------------
print("\n--- 5.3 Array Reshaping for ML ---")

# Flatten image-like data
image = np.random.random((28, 28))
flattened = image.flatten()
print(f"Image shape: {image.shape}")
print(f"Flattened shape: {flattened.shape}")

# Reshape for batch processing
data = np.random.random((100, 784))  # 100 samples, 784 features
batch = data.reshape(10, 10, 784)  # 10 batches of 10 samples
print(f"\nData shape: {data.shape}")
print(f"Batch shape: {batch.shape}")


# ============================================================================
# 6. COMMON PATTERNS AND BEST PRACTICES
# ============================================================================

print("\n" + "=" * 70)
print("6. COMMON PATTERNS AND BEST PRACTICES")
print("=" * 70)

print("""
1. ARRAY CREATION:
   - Use np.array() for lists
   - Use np.zeros(), np.ones() for initialization
   - Use np.arange() or np.linspace() for sequences
   - Specify dtype when needed

2. INDEXING:
   - Use boolean indexing for filtering
   - Use fancy indexing for specific elements
   - Remember slicing creates views (not copies)

3. OPERATIONS:
   - Element-wise operations are automatic
   - Use broadcasting for different shapes
   - Use np.dot() or @ for matrix multiplication
   - Use ufuncs for mathematical operations

4. PERFORMANCE:
   - NumPy operations are vectorized (fast)
   - Avoid Python loops when possible
   - Use boolean indexing instead of loops
   - Prefer NumPy functions over Python equivalents

5. MEMORY:
   - Slicing creates views (shares memory)
   - Use .copy() for independent copies
   - Reshape doesn't copy data (creates view)
   - Flatten creates copy, ravel creates view

6. DEBUGGING:
   - Check shape with .shape
   - Check dtype with .dtype
   - Use print() to inspect arrays
   - Verify broadcasting compatibility
""")


# ============================================================================
# 7. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Create and manipulate arrays
print("\n--- Exercise 1: Create and Manipulate Arrays ---")
arr = np.array([1, 2, 3, 4, 5])
print(f"Original: {arr}")
print(f"Square each element: {arr ** 2}")
print(f"Add 10 to each: {arr + 10}")

# Exercise 2: Filter array
print("\n--- Exercise 2: Filter Array ---")
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
evens = arr[arr % 2 == 0]
print(f"Even numbers: {evens}")

# Exercise 3: Reshape array
print("\n--- Exercise 3: Reshape Array ---")
arr = np.arange(12)
print(f"Original (1D): {arr}")
reshaped = arr.reshape(3, 4)
print(f"Reshaped (3x4):\n{reshaped}")

# Exercise 4: Broadcasting
print("\n--- Exercise 4: Broadcasting ---")
arr = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
result = arr + row
print(f"Array:\n{arr}")
print(f"Row: {row}")
print(f"Result:\n{result}")


# ============================================================================
# 8. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("8. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. ARRAY CREATION:
   - np.array() from lists
   - np.zeros(), np.ones(), np.full() for initialization
   - np.arange(), np.linspace() for sequences
   - Know shape, dtype, size properties

2. INDEXING:
   - Basic: arr[i], arr[i, j]
   - Slicing: arr[start:stop:step]
   - Boolean: arr[arr > 5]
   - Fancy: arr[[0, 2, 4]]

3. ARRAY METHODS:
   - shape: dimensions (rows, cols)
   - dtype: data type
   - reshape(): change shape
   - astype(): convert type
   - mean(), sum(), min(), max()

4. OPERATIONS:
   - Element-wise: automatic with +, -, *, /
   - Broadcasting: operations on different shapes
   - Matrix multiplication: np.dot() or @
   - Universal functions: np.sqrt(), np.sin(), etc.

5. BROADCASTING RULES:
   - Align from right
   - Dimensions match or are 1
   - Missing dimensions treated as 1
   - Understand when it works and when it fails

6. COMMON USE CASES:
   - Filtering: arr[arr > threshold]
   - Reshaping: arr.reshape(-1, 1) for ML
   - Aggregation: arr.mean(axis=0)
   - Vectorized operations instead of loops

7. INTERVIEW TIPS:
   - Explain broadcasting concept
   - Know difference between * and @
   - Understand shape compatibility
   - Use vectorized operations
   - Know when to use boolean indexing
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NumPy Basics Guide Ready!")
    print("=" * 70)
