"""
Python Lists - Interview Preparation
Topic 2.1: Lists

This module covers:
- Creation: List literals, list(), list comprehensions
- Indexing & Slicing: Positive/negative indices, slice notation
- Methods: append, extend, insert, remove, pop, index, count, sort, reverse
- List Comprehensions: Basic, nested, conditional
- Iteration: for loops, enumerate(), zip()
"""

# ============================================================================
# 1. LIST CREATION
# ============================================================================

print("=" * 70)
print("1. LIST CREATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 List Literals
# ----------------------------------------------------------------------------
print("\n--- List Literals ---")

# Empty list
empty_list = []
print(f"Empty list: {empty_list}")

# List with elements
numbers = [1, 2, 3, 4, 5]
print(f"Numbers: {numbers}")

# Mixed types
mixed = [1, "hello", 3.14, True, None]
print(f"Mixed types: {mixed}")

# Nested lists
nested = [[1, 2], [3, 4], [5, 6]]
print(f"Nested list: {nested}")

# List with duplicates
duplicates = [1, 2, 2, 3, 3, 3]
print(f"List with duplicates: {duplicates}")


# ----------------------------------------------------------------------------
# 1.2 Using list() Constructor
# ----------------------------------------------------------------------------
print("\n--- Using list() Constructor ---")

# From string (iterable)
chars = list("hello")
print(f"list('hello'): {chars}")

# From range
numbers = list(range(5))
print(f"list(range(5)): {numbers}")

# From tuple
tuple_data = (1, 2, 3)
list_from_tuple = list(tuple_data)
print(f"list((1, 2, 3)): {list_from_tuple}")

# From another list (creates shallow copy)
original = [1, 2, 3]
copied = list(original)
print(f"Original: {original}, Copied: {copied}")
print(f"Same object? {original is copied}")  # False - different objects


# ----------------------------------------------------------------------------
# 1.3 List Comprehensions (Creation)
# ----------------------------------------------------------------------------
print("\n--- List Comprehensions (Creation) ---")

# Basic list comprehension
squares = [x ** 2 for x in range(5)]
print(f"Squares: {squares}")

# With condition
evens = [x for x in range(10) if x % 2 == 0]
print(f"Even numbers: {evens}")

# Transforming elements
words = ["hello", "world", "python"]
lengths = [len(word) for word in words]
print(f"Word lengths: {lengths}")

# More examples in section 4


# ============================================================================
# 2. INDEXING & SLICING
# ============================================================================

print("\n" + "=" * 70)
print("2. INDEXING & SLICING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Positive Indices
# ----------------------------------------------------------------------------
print("\n--- Positive Indices ---")

numbers = [10, 20, 30, 40, 50]
print(f"List: {numbers}")

# Access first element (index 0)
print(f"numbers[0] = {numbers[0]}")

# Access second element (index 1)
print(f"numbers[1] = {numbers[1]}")

# Access last element (using len)
print(f"numbers[len(numbers)-1] = {numbers[len(numbers)-1]}")

# Access last element (using -1, see negative indices)
print(f"numbers[-1] = {numbers[-1]}")


# ----------------------------------------------------------------------------
# 2.2 Negative Indices
# ----------------------------------------------------------------------------
print("\n--- Negative Indices ---")

numbers = [10, 20, 30, 40, 50]
print(f"List: {numbers}")

# Negative indices count from the end
print(f"numbers[-1] = {numbers[-1]}")  # Last element
print(f"numbers[-2] = {numbers[-2]}")  # Second to last
print(f"numbers[-3] = {numbers[-3]}")  # Third to last

# Index mapping
# Index:  0   1   2   3   4
# Value: 10  20  30  40  50
# Neg:   -5  -4  -3  -2  -1


# ----------------------------------------------------------------------------
# 2.3 Index Errors
# ----------------------------------------------------------------------------
print("\n--- Index Errors ---")

numbers = [10, 20, 30]

# Valid indices
print(f"numbers[0] = {numbers[0]}")
print(f"numbers[-1] = {numbers[-1]}")

# Invalid indices (will raise IndexError)
# numbers[5]  # IndexError: list index out of range
# numbers[-5]  # IndexError: list index out of range

# Safe access with bounds checking
def safe_get(lst, index):
    """Safely get element at index."""
    if 0 <= index < len(lst) or -len(lst) <= index < 0:
        return lst[index]
    return None

print(f"safe_get(numbers, 1) = {safe_get(numbers, 1)}")
print(f"safe_get(numbers, 10) = {safe_get(numbers, 10)}")


# ----------------------------------------------------------------------------
# 2.4 Basic Slicing
# ----------------------------------------------------------------------------
print("\n--- Basic Slicing ---")

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Original list: {numbers}")

# Syntax: list[start:stop]
# Returns elements from index start (inclusive) to stop (exclusive)

# Get first 3 elements
print(f"numbers[0:3] = {numbers[0:3]}")  # [0, 1, 2]

# Get elements from index 2 to 5
print(f"numbers[2:5] = {numbers[2:5]}")  # [2, 3, 4]

# Start defaults to 0
print(f"numbers[:3] = {numbers[:3]}")  # [0, 1, 2]

# Stop defaults to end
print(f"numbers[3:] = {numbers[3:]}")  # [3, 4, 5, 6, 7, 8, 9]

# Both defaults (copy entire list)
print(f"numbers[:] = {numbers[:]}")  # Full copy


# ----------------------------------------------------------------------------
# 2.5 Slicing with Step
# ----------------------------------------------------------------------------
print("\n--- Slicing with Step ---")

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Original list: {numbers}")

# Syntax: list[start:stop:step]
# Step determines the increment

# Every second element
print(f"numbers[::2] = {numbers[::2]}")  # [0, 2, 4, 6, 8]

# Every third element
print(f"numbers[::3] = {numbers[::3]}")  # [0, 3, 6, 9]

# From index 1, every second element
print(f"numbers[1::2] = {numbers[1::2]}")  # [1, 3, 5, 7, 9]

# Reverse list (negative step)
print(f"numbers[::-1] = {numbers[::-1]}")  # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# Every second element in reverse
print(f"numbers[::-2] = {numbers[::-2]}")  # [9, 7, 5, 3, 1]


# ----------------------------------------------------------------------------
# 2.6 Negative Indices in Slicing
# ----------------------------------------------------------------------------
print("\n--- Negative Indices in Slicing ---")

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Original list: {numbers}")

# Last 3 elements
print(f"numbers[-3:] = {numbers[-3:]}")  # [7, 8, 9]

# All except last 3
print(f"numbers[:-3] = {numbers[:-3]}")  # [0, 1, 2, 3, 4, 5, 6]

# From -5 to -2
print(f"numbers[-5:-2] = {numbers[-5:-2]}")  # [5, 6, 7]

# Last 5 elements, every second
print(f"numbers[-5::2] = {numbers[-5::2]}")  # [5, 7, 9]


# ----------------------------------------------------------------------------
# 2.7 Slicing Examples
# ----------------------------------------------------------------------------
print("\n--- Slicing Examples ---")

text = list("Python")
print(f"Text: {text}")

# Get first 3 characters
print(f"First 3: {text[:3]}")  # ['P', 'y', 't']

# Get last 3 characters
print(f"Last 3: {text[-3:]}")  # ['h', 'o', 'n']

# Get middle characters
print(f"Middle: {text[2:4]}")  # ['t', 'h']

# Reverse string
print(f"Reversed: {text[::-1]}")  # ['n', 'o', 'h', 't', 'y', 'P']


# ----------------------------------------------------------------------------
# 2.8 Modifying Elements via Indexing
# ----------------------------------------------------------------------------
print("\n--- Modifying Elements via Indexing ---")

numbers = [1, 2, 3, 4, 5]
print(f"Original: {numbers}")

# Modify element at index
numbers[0] = 10
print(f"After numbers[0] = 10: {numbers}")

# Modify multiple elements
numbers[1:3] = [20, 30]
print(f"After numbers[1:3] = [20, 30]: {numbers}")

# Replace with different size
numbers[2:4] = [100]
print(f"After numbers[2:4] = [100]: {numbers}")

# Insert elements
numbers[1:1] = [50, 60]
print(f"After numbers[1:1] = [50, 60]: {numbers}")


# ============================================================================
# 3. LIST METHODS
# ============================================================================

print("\n" + "=" * 70)
print("3. LIST METHODS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 append() - Add Element to End
# ----------------------------------------------------------------------------
print("\n--- append() - Add Element to End ---")

numbers = [1, 2, 3]
print(f"Original: {numbers}")

numbers.append(4)
print(f"After append(4): {numbers}")

numbers.append(5)
print(f"After append(5): {numbers}")

# Append single element (not list)
numbers.append([6, 7])  # Adds list as single element
print(f"After append([6, 7]): {numbers}")


# ----------------------------------------------------------------------------
# 3.2 extend() - Add Multiple Elements
# ----------------------------------------------------------------------------
print("\n--- extend() - Add Multiple Elements ---")

numbers = [1, 2, 3]
print(f"Original: {numbers}")

numbers.extend([4, 5])
print(f"After extend([4, 5]): {numbers}")

numbers.extend([6, 7, 8])
print(f"After extend([6, 7, 8]): {numbers}")

# extend() with any iterable
numbers.extend(range(9, 12))
print(f"After extend(range(9, 12)): {numbers}")


# ----------------------------------------------------------------------------
# 3.3 insert() - Insert at Specific Position
# ----------------------------------------------------------------------------
print("\n--- insert() - Insert at Specific Position ---")

numbers = [1, 2, 3, 4]
print(f"Original: {numbers}")

# Insert at index 0 (beginning)
numbers.insert(0, 0)
print(f"After insert(0, 0): {numbers}")

# Insert at index 2
numbers.insert(2, 1.5)
print(f"After insert(2, 1.5): {numbers}")

# Insert at end (same as append)
numbers.insert(len(numbers), 5)
print(f"After insert(len(numbers), 5): {numbers}")


# ----------------------------------------------------------------------------
# 3.4 remove() - Remove First Occurrence
# ----------------------------------------------------------------------------
print("\n--- remove() - Remove First Occurrence ---")

numbers = [1, 2, 3, 2, 4, 2]
print(f"Original: {numbers}")

numbers.remove(2)  # Removes first occurrence
print(f"After remove(2): {numbers}")

numbers.remove(2)  # Removes next occurrence
print(f"After remove(2) again: {numbers}")

# ValueError if element not found
# numbers.remove(99)  # ValueError: list.remove(x): x not in list


# ----------------------------------------------------------------------------
# 3.5 pop() - Remove and Return Element
# ----------------------------------------------------------------------------
print("\n--- pop() - Remove and Return Element ---")

numbers = [1, 2, 3, 4, 5]
print(f"Original: {numbers}")

# pop() without index (removes last)
last = numbers.pop()
print(f"After pop(): {numbers}, returned: {last}")

# pop() with index
second = numbers.pop(1)
print(f"After pop(1): {numbers}, returned: {second}")

# pop() at beginning
first = numbers.pop(0)
print(f"After pop(0): {numbers}, returned: {first}")

# IndexError if index out of range
# numbers.pop(10)  # IndexError: pop index out of range


# ----------------------------------------------------------------------------
# 3.6 index() - Find Index of Element
# ----------------------------------------------------------------------------
print("\n--- index() - Find Index of Element ---")

numbers = [10, 20, 30, 20, 40]
print(f"List: {numbers}")

# Find first occurrence
idx = numbers.index(20)
print(f"index(20) = {idx}")

# Find with start position
idx = numbers.index(20, 2)  # Start searching from index 2
print(f"index(20, 2) = {idx}")

# Find with start and stop
idx = numbers.index(20, 1, 3)  # Search between indices 1 and 3
print(f"index(20, 1, 3) = {idx}")

# ValueError if element not found
# numbers.index(99)  # ValueError: 99 is not in list


# ----------------------------------------------------------------------------
# 3.7 count() - Count Occurrences
# ----------------------------------------------------------------------------
print("\n--- count() - Count Occurrences ---")

numbers = [1, 2, 3, 2, 4, 2, 5]
print(f"List: {numbers}")

count_2 = numbers.count(2)
print(f"count(2) = {count_2}")

count_7 = numbers.count(7)
print(f"count(7) = {count_7}")  # Returns 0 if not found

# Count in mixed list
mixed = [1, "hello", 1, True, 1, None]
count_1 = mixed.count(1)
print(f"count(1) in {mixed} = {count_1}")  # Counts both int 1 and True


# ----------------------------------------------------------------------------
# 3.8 sort() - Sort List In-Place
# ----------------------------------------------------------------------------
print("\n--- sort() - Sort List In-Place ---")

numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"Original: {numbers}")

# Sort ascending (default)
numbers.sort()
print(f"After sort(): {numbers}")

# Sort descending
numbers.sort(reverse=True)
print(f"After sort(reverse=True): {numbers}")

# Sort strings
words = ["banana", "apple", "cherry"]
words.sort()
print(f"Sorted words: {words}")

# Sort with key function
words = ["apple", "banana", "cherry", "date"]
words.sort(key=len)  # Sort by length
print(f"Sorted by length: {words}")

# Sort with lambda
students = [("Alice", 25), ("Bob", 20), ("Charlie", 22)]
students.sort(key=lambda x: x[1])  # Sort by age
print(f"Sorted by age: {students}")


# ----------------------------------------------------------------------------
# 3.9 reverse() - Reverse List In-Place
# ----------------------------------------------------------------------------
print("\n--- reverse() - Reverse List In-Place ---")

numbers = [1, 2, 3, 4, 5]
print(f"Original: {numbers}")

numbers.reverse()
print(f"After reverse(): {numbers}")

# Note: reverse() modifies in-place, returns None
result = numbers.reverse()
print(f"Return value of reverse(): {result}")


# ----------------------------------------------------------------------------
# 3.10 Other Useful Methods
# ----------------------------------------------------------------------------
print("\n--- Other Useful Methods ---")

numbers = [1, 2, 3]

# clear() - Remove all elements
numbers_copy = numbers.copy()
numbers_copy.clear()
print(f"After clear(): {numbers_copy}")

# copy() - Shallow copy
original = [1, 2, [3, 4]]
copied = original.copy()
print(f"Original: {original}, Copied: {copied}")
print(f"Same object? {original is copied}")  # False
print(f"Nested list same? {original[2] is copied[2]}")  # True (shallow copy)

# len() - Get length
print(f"Length of {numbers}: {len(numbers)}")

# in operator - Check membership
print(f"2 in {numbers}: {2 in numbers}")
print(f"5 in {numbers}: {5 in numbers}")


# ============================================================================
# 4. LIST COMPREHENSIONS
# ============================================================================

print("\n" + "=" * 70)
print("4. LIST COMPREHENSIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Basic List Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Basic List Comprehensions ---")

# Syntax: [expression for item in iterable]

# Square numbers
squares = [x ** 2 for x in range(5)]
print(f"Squares: {squares}")

# Double numbers
numbers = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in numbers]
print(f"Doubled: {doubled}")

# Convert to strings
strings = [str(x) for x in range(5)]
print(f"Strings: {strings}")


# ----------------------------------------------------------------------------
# 4.2 List Comprehensions with Conditions
# ----------------------------------------------------------------------------
print("\n--- List Comprehensions with Conditions ---")

# Syntax: [expression for item in iterable if condition]

# Even numbers
evens = [x for x in range(10) if x % 2 == 0]
print(f"Even numbers: {evens}")

# Numbers greater than 5
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
greater_than_5 = [x for x in numbers if x > 5]
print(f"Numbers > 5: {greater_than_5}")

# Words longer than 3 characters
words = ["apple", "pie", "banana", "cat", "cherry"]
long_words = [word for word in words if len(word) > 3]
print(f"Long words: {long_words}")


# ----------------------------------------------------------------------------
# 4.3 List Comprehensions with if-else
# ----------------------------------------------------------------------------
print("\n--- List Comprehensions with if-else ---")

# Syntax: [expression1 if condition else expression2 for item in iterable]

# Mark even/odd
numbers = [1, 2, 3, 4, 5, 6]
parity = ["even" if x % 2 == 0 else "odd" for x in numbers]
print(f"Parity: {parity}")

# Square evens, cube odds
transformed = [x ** 2 if x % 2 == 0 else x ** 3 for x in range(1, 6)]
print(f"Transformed: {transformed}")


# ----------------------------------------------------------------------------
# 4.4 Nested List Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Nested List Comprehensions ---")

# Flatten nested list
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
print(f"Flattened: {flattened}")

# Multiplication table
table = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(f"Multiplication table: {table}")

# Cartesian product
colors = ["red", "blue"]
sizes = ["S", "M", "L"]
combinations = [(color, size) for color in colors for size in sizes]
print(f"Combinations: {combinations}")


# ----------------------------------------------------------------------------
# 4.5 List Comprehensions vs Loops
# ----------------------------------------------------------------------------
print("\n--- List Comprehensions vs Loops ---")

# Using loop
squares_loop = []
for x in range(5):
    squares_loop.append(x ** 2)
print(f"Using loop: {squares_loop}")

# Using list comprehension (more Pythonic)
squares_comp = [x ** 2 for x in range(5)]
print(f"Using comprehension: {squares_comp}")

# List comprehensions are generally more readable and often faster


# ----------------------------------------------------------------------------
# 4.6 Complex List Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Complex List Comprehensions ---")

# Multiple conditions
numbers = range(20)
filtered = [x for x in numbers if x % 2 == 0 if x % 3 == 0]
print(f"Divisible by 2 and 3: {filtered}")

# Nested with condition
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
evens = [num for row in matrix for num in row if num % 2 == 0]
print(f"Even numbers in matrix: {evens}")


# ============================================================================
# 5. ITERATION
# ============================================================================

print("\n" + "=" * 70)
print("5. ITERATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Basic for Loop
# ----------------------------------------------------------------------------
print("\n--- Basic for Loop ---")

numbers = [1, 2, 3, 4, 5]

# Iterate over elements
print("Iterating over elements:")
for num in numbers:
    print(f"  {num}")

# Iterate with index (manual)
print("\nIterating with index (manual):")
for i in range(len(numbers)):
    print(f"  Index {i}: {numbers[i]}")


# ----------------------------------------------------------------------------
# 5.2 enumerate() - Get Index and Value
# ----------------------------------------------------------------------------
print("\n--- enumerate() - Get Index and Value ---")

fruits = ["apple", "banana", "cherry"]

# Basic enumerate
print("Using enumerate():")
for index, fruit in enumerate(fruits):
    print(f"  Index {index}: {fruit}")

# Start index from 1
print("\nStarting from 1:")
for index, fruit in enumerate(fruits, start=1):
    print(f"  Position {index}: {fruit}")

# Convert to list
indexed = list(enumerate(fruits))
print(f"\nAs list: {indexed}")


# ----------------------------------------------------------------------------
# 5.3 zip() - Iterate Multiple Lists
# ----------------------------------------------------------------------------
print("\n--- zip() - Iterate Multiple Lists ---")

names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
cities = ["New York", "London", "Tokyo"]

# Zip two lists
print("Zipping two lists:")
for name, age in zip(names, ages):
    print(f"  {name} is {age} years old")

# Zip multiple lists
print("\nZipping multiple lists:")
for name, age, city in zip(names, ages, cities):
    print(f"  {name}, {age}, from {city}")

# Convert to list
zipped = list(zip(names, ages))
print(f"\nAs list: {zipped}")

# Unzipping
pairs = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
unzipped_names, unzipped_ages = zip(*pairs)
print(f"Unzipped names: {unzipped_names}")
print(f"Unzipped ages: {unzipped_ages}")


# ----------------------------------------------------------------------------
# 5.4 zip() with Different Lengths
# ----------------------------------------------------------------------------
print("\n--- zip() with Different Lengths ---")

list1 = [1, 2, 3, 4]
list2 = ["a", "b", "c"]

# zip() stops at shortest list
zipped = list(zip(list1, list2))
print(f"zip([1,2,3,4], ['a','b','c']): {zipped}")

# Using itertools.zip_longest for different behavior
from itertools import zip_longest
zipped_long = list(zip_longest(list1, list2, fillvalue=None))
print(f"zip_longest: {zipped_long}")


# ----------------------------------------------------------------------------
# 5.5 Iterating with Conditions
# ----------------------------------------------------------------------------
print("\n--- Iterating with Conditions ---")

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Iterate and filter
print("Even numbers:")
for num in numbers:
    if num % 2 == 0:
        print(f"  {num}")

# Using enumerate with condition
print("\nEven numbers with index:")
for index, num in enumerate(numbers):
    if num % 2 == 0:
        print(f"  Index {index}: {num}")


# ----------------------------------------------------------------------------
# 5.6 Iterating Nested Lists
# ----------------------------------------------------------------------------
print("\n--- Iterating Nested Lists ---")

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Iterate rows
print("By rows:")
for row in matrix:
    print(f"  {row}")

# Iterate elements
print("\nAll elements:")
for row in matrix:
    for element in row:
        print(f"  {element}", end=" ")
print()

# With indices
print("\nWith indices:")
for i, row in enumerate(matrix):
    for j, element in enumerate(row):
        print(f"  matrix[{i}][{j}] = {element}")


# ----------------------------------------------------------------------------
# 5.7 Iterating with Slicing
# ----------------------------------------------------------------------------
print("\n--- Iterating with Slicing ---")

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Iterate first 5 elements
print("First 5 elements:")
for num in numbers[:5]:
    print(f"  {num}")

# Iterate every second element
print("\nEvery second element:")
for num in numbers[::2]:
    print(f"  {num}", end=" ")
print()

# Iterate in reverse
print("\nIn reverse:")
for num in numbers[::-1]:
    print(f"  {num}", end=" ")
print()


# ============================================================================
# 6. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Find maximum in list
print("\n--- Exercise 1: Find Maximum ---")
def find_max(numbers):
    """Find maximum value in list."""
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val

numbers = [3, 7, 2, 9, 1, 5]
print(f"Maximum of {numbers}: {find_max(numbers)}")


# Exercise 2: Remove duplicates
print("\n--- Exercise 2: Remove Duplicates ---")
def remove_duplicates(lst):
    """Remove duplicates while preserving order."""
    seen = []
    result = []
    for item in lst:
        if item not in seen:
            seen.append(item)
            result.append(item)
    return result

data = [1, 2, 2, 3, 3, 3, 4, 5, 5]
print(f"Remove duplicates from {data}: {remove_duplicates(data)}")


# Exercise 3: Rotate list
print("\n--- Exercise 3: Rotate List ---")
def rotate_list(lst, k):
    """Rotate list k positions to the right."""
    if not lst:
        return lst
    k = k % len(lst)  # Handle k > len(lst)
    return lst[-k:] + lst[:-k]

numbers = [1, 2, 3, 4, 5]
print(f"Rotate {numbers} by 2: {rotate_list(numbers, 2)}")


# Exercise 4: Flatten nested list
print("\n--- Exercise 4: Flatten Nested List ---")
def flatten(nested):
    """Flatten nested list recursively."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

nested = [[1, 2], [3, [4, 5]], 6, [7, 8, 9]]
print(f"Flatten {nested}: {flatten(nested)}")


# Exercise 5: List comprehension - squares of evens
print("\n--- Exercise 5: Squares of Even Numbers ---")
numbers = range(10)
squared_evens = [x ** 2 for x in numbers if x % 2 == 0]
print(f"Squares of evens: {squared_evens}")


# Exercise 6: Find common elements
print("\n--- Exercise 6: Find Common Elements ---")
def find_common(list1, list2):
    """Find common elements between two lists."""
    return [item for item in list1 if item in list2]

list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]
print(f"Common elements: {find_common(list1, list2)}")


# Exercise 7: Group by length
print("\n--- Exercise 7: Group Words by Length ---")
def group_by_length(words):
    """Group words by their length."""
    groups = {}
    for word in words:
        length = len(word)
        if length not in groups:
            groups[length] = []
        groups[length].append(word)
    return groups

words = ["apple", "pie", "banana", "cat", "cherry", "dog"]
print(f"Grouped by length: {group_by_length(words)}")


# Exercise 8: Matrix transpose
print("\n--- Exercise 8: Matrix Transpose ---")
def transpose(matrix):
    """Transpose a matrix."""
    if not matrix:
        return []
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(f"Original: {matrix}")
print(f"Transposed: {transpose(matrix)}")


# ============================================================================
# 7. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("7. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between append() and extend()?
print("\n--- Q1: append() vs extend() ---")
print("""
append(): Adds single element (can be list, but adds as one element)
extend(): Adds all elements from iterable (unpacks iterable)
""")

# Q2: What's the difference between list() and []?
print("\n--- Q2: list() vs [] ---")
print("""
[]: List literal, creates list directly
list(): Constructor, can convert from other iterables
Both create new list objects
""")

# Q3: How does slicing work?
print("\n--- Q3: Slicing ---")
print("""
Syntax: list[start:stop:step]
- start: inclusive (default 0)
- stop: exclusive (default end)
- step: increment (default 1)
- Negative indices count from end
- Returns new list (doesn't modify original)
""")

# Q4: What's the difference between sort() and sorted()?
print("\n--- Q4: sort() vs sorted() ---")
print("""
sort(): Modifies list in-place, returns None
sorted(): Returns new sorted list, doesn't modify original
Use sort() when you want to modify, sorted() when you want to keep original
""")

# Q5: How to reverse a list?
print("\n--- Q5: Reversing a List ---")
print("""
1. list.reverse() - modifies in-place
2. list[::-1] - creates new reversed list
3. reversed(list) - returns iterator
4. list.sort(reverse=True) - if you want sorted reverse
""")

# Q6: What are list comprehensions?
print("\n--- Q6: List Comprehensions ---")
print("""
List comprehensions: Concise way to create lists
Syntax: [expression for item in iterable if condition]
More Pythonic and often faster than loops
Can be nested for complex operations
""")

# Q7: How does enumerate() work?
print("\n--- Q7: enumerate() ---")
print("""
enumerate(): Returns (index, value) pairs
Useful when you need both index and value
Can specify start index with start parameter
More Pythonic than range(len(list))
""")


# ============================================================================
# 8. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("8. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. LIST CREATION:
   - Use [] for empty list or list literals
   - Use list() constructor to convert from iterables
   - List comprehensions are Pythonic way to create lists

2. INDEXING:
   - Positive indices: 0 to len-1 (left to right)
   - Negative indices: -1 to -len (right to left)
   - IndexError if index out of range

3. SLICING:
   - Syntax: list[start:stop:step]
   - start inclusive, stop exclusive
   - Returns new list (doesn't modify original)
   - Negative step reverses order
   - list[:] creates shallow copy

4. LIST METHODS:
   - append(): Add single element to end
   - extend(): Add multiple elements
   - insert(): Insert at specific position
   - remove(): Remove first occurrence
   - pop(): Remove and return element
   - index(): Find index of element
   - count(): Count occurrences
   - sort(): Sort in-place
   - reverse(): Reverse in-place

5. LIST COMPREHENSIONS:
   - More Pythonic than loops
   - Syntax: [expr for item in iterable if condition]
   - Can be nested for complex operations
   - Often faster than equivalent loops

6. ITERATION:
   - Use for item in list for simple iteration
   - Use enumerate() when you need index
   - Use zip() to iterate multiple lists
   - Can combine with slicing and conditions

7. COMMON PATTERNS:
   - list[:] for shallow copy
   - list[::-1] for reverse copy
   - [x for x in list if condition] for filtering
   - [f(x) for x in list] for transformation

8. BEST PRACTICES:
   - Use list comprehensions when possible
   - Prefer extend() over multiple append()
   - Use enumerate() instead of range(len())
   - Be careful with mutable default arguments
   - Remember slicing creates new list
   - Use in operator for membership testing
""")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
