"""
Searching & Sorting - Interview Preparation
Topic 6.2: Searching & Sorting

This module covers:
- Linear Search: O(n) search
- Binary Search: O(log n) search on sorted arrays
- Built-in Sorting: sorted(), list.sort()
- Custom Sorting: key parameter, lambda functions
"""

from typing import List, Optional, Callable

# ============================================================================
# 1. LINEAR SEARCH
# ============================================================================

print("=" * 70)
print("1. LINEAR SEARCH")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Basic Linear Search
# ----------------------------------------------------------------------------
print("\n--- 1.1 Basic Linear Search ---")

def linear_search(arr: List, target) -> int:
    """
    Linear search - find index of target in array.
    Time: O(n), Space: O(1)
    Returns -1 if not found.
    """
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1

# Example
arr = [5, 2, 8, 1, 9, 3]
target = 8
result = linear_search(arr, target)
print(f"Array: {arr}, Target: {target}")
print(f"Found at index: {result}")


def linear_search_all(arr: List, target) -> List[int]:
    """
    Find all indices where target appears.
    Time: O(n), Space: O(k) where k is number of occurrences
    """
    indices = []
    for i, item in enumerate(arr):
        if item == target:
            indices.append(i)
    return indices

# Example
arr = [5, 2, 8, 2, 9, 2]
target = 2
result = linear_search_all(arr, target)
print(f"\nArray: {arr}, Target: {target}")
print(f"All indices: {result}")


# ----------------------------------------------------------------------------
# 1.2 Linear Search with Condition
# ----------------------------------------------------------------------------
print("\n--- 1.2 Linear Search with Condition ---")

def linear_search_condition(arr: List[int], condition: Callable) -> Optional[int]:
    """
    Find first element satisfying condition.
    Time: O(n), Space: O(1)
    """
    for i, item in enumerate(arr):
        if condition(item):
            return i
    return None

# Example
arr = [5, 2, 8, 1, 9, 3]
# Find first even number
result = linear_search_condition(arr, lambda x: x % 2 == 0)
print(f"Array: {arr}")
print(f"First even number at index: {result}")


def find_max_linear(arr: List[int]) -> tuple:
    """
    Find maximum value and its index using linear search.
    Time: O(n), Space: O(1)
    """
    if not arr:
        return None, -1
    
    max_val = arr[0]
    max_idx = 0
    
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    
    return max_val, max_idx

# Example
arr = [5, 2, 8, 1, 9, 3]
max_val, max_idx = find_max_linear(arr)
print(f"\nArray: {arr}")
print(f"Maximum value: {max_val} at index: {max_idx}")


# ----------------------------------------------------------------------------
# 1.3 Linear Search Applications
# ----------------------------------------------------------------------------
print("\n--- 1.3 Linear Search Applications ---")

def find_first_occurrence(arr: List, target) -> int:
    """Find first occurrence of target."""
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1

def find_last_occurrence(arr: List, target) -> int:
    """Find last occurrence of target."""
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] == target:
            return i
    return -1

# Example
arr = [1, 2, 3, 2, 4, 2, 5]
target = 2
print(f"Array: {arr}, Target: {target}")
print(f"First occurrence: {find_first_occurrence(arr, target)}")
print(f"Last occurrence: {find_last_occurrence(arr, target)}")


# ============================================================================
# 2. BINARY SEARCH
# ============================================================================

print("\n" + "=" * 70)
print("2. BINARY SEARCH")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic Binary Search
# ----------------------------------------------------------------------------
print("\n--- 2.1 Basic Binary Search ---")

def binary_search(arr: List[int], target: int) -> int:
    """
    Binary search in sorted array.
    Time: O(log n), Space: O(1)
    Returns index if found, -1 otherwise.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Example
arr = [1, 2, 3, 5, 8, 9, 11, 13, 15]
target = 9
result = binary_search(arr, target)
print(f"Sorted array: {arr}, Target: {target}")
print(f"Found at index: {result}")

target = 10
result = binary_search(arr, target)
print(f"\nTarget: {target}, Found at index: {result}")


# ----------------------------------------------------------------------------
# 2.2 Binary Search - Recursive Version
# ----------------------------------------------------------------------------
print("\n--- 2.2 Binary Search - Recursive Version ---")

def binary_search_recursive(arr: List[int], target: int, left: int = 0, right: int = None) -> int:
    """
    Recursive binary search.
    Time: O(log n), Space: O(log n) due to recursion stack
    """
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Example
arr = [1, 2, 3, 5, 8, 9, 11, 13, 15]
target = 5
result = binary_search_recursive(arr, target)
print(f"Array: {arr}, Target: {target}")
print(f"Found at index: {result}")


# ----------------------------------------------------------------------------
# 2.3 Binary Search Variations
# ----------------------------------------------------------------------------
print("\n--- 2.3 Binary Search Variations ---")

def binary_search_first_occurrence(arr: List[int], target: int) -> int:
    """
    Find first occurrence of target in sorted array (may have duplicates).
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Example
arr = [1, 2, 2, 2, 3, 4, 5]
target = 2
result = binary_search_first_occurrence(arr, target)
print(f"Array: {arr}, Target: {target}")
print(f"First occurrence at index: {result}")


def binary_search_last_occurrence(arr: List[int], target: int) -> int:
    """
    Find last occurrence of target in sorted array.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Example
arr = [1, 2, 2, 2, 3, 4, 5]
target = 2
result = binary_search_last_occurrence(arr, target)
print(f"\nArray: {arr}, Target: {target}")
print(f"Last occurrence at index: {result}")


def binary_search_insert_position(arr: List[int], target: int) -> int:
    """
    Find insertion position for target in sorted array.
    Returns index where target should be inserted.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

# Example
arr = [1, 3, 5, 7, 9]
target = 6
result = binary_search_insert_position(arr, target)
print(f"\nArray: {arr}, Target: {target}")
print(f"Insert position: {result}")


# ----------------------------------------------------------------------------
# 2.4 Binary Search on Answer Space
# ----------------------------------------------------------------------------
print("\n--- 2.4 Binary Search on Answer Space ---")

def sqrt_binary_search(x: int) -> int:
    """
    Find integer square root using binary search.
    Time: O(log x), Space: O(1)
    """
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Return floor of square root

# Example
for x in [4, 8, 16, 25]:
    result = sqrt_binary_search(x)
    print(f"sqrt({x}) ≈ {result}")


# ============================================================================
# 3. BUILT-IN SORTING
# ============================================================================

print("\n" + "=" * 70)
print("3. BUILT-IN SORTING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 sorted() Function
# ----------------------------------------------------------------------------
print("\n--- 3.1 sorted() Function ---")

# sorted() returns new sorted list
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"Original: {numbers}")
sorted_numbers = sorted(numbers)
print(f"sorted(numbers): {sorted_numbers}")
print(f"Original unchanged: {numbers}")

# Sort descending
sorted_desc = sorted(numbers, reverse=True)
print(f"\nsorted(numbers, reverse=True): {sorted_desc}")

# Sort strings
words = ["banana", "apple", "cherry"]
sorted_words = sorted(words)
print(f"\nWords: {words}")
print(f"Sorted: {sorted_words}")

# Sort different types
mixed = [3, "apple", 1, "banana", 2]
# sorted(mixed)  # Would raise TypeError - can't compare different types


# ----------------------------------------------------------------------------
# 3.2 list.sort() Method
# ----------------------------------------------------------------------------
print("\n--- 3.2 list.sort() Method ---")

# list.sort() modifies list in-place
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"Original: {numbers}")
numbers.sort()
print(f"After sort(): {numbers}")

# Sort descending
numbers.sort(reverse=True)
print(f"After sort(reverse=True): {numbers}")

# Note: sort() returns None
result = numbers.sort()
print(f"Return value of sort(): {result}")


# ----------------------------------------------------------------------------
# 3.3 sorted() vs list.sort()
# ----------------------------------------------------------------------------
print("\n--- 3.3 sorted() vs list.sort() ---")
print("""
sorted():
- Returns new sorted list
- Original list unchanged
- Can sort any iterable
- Returns list

list.sort():
- Modifies list in-place
- Returns None
- Only works on lists
- Slightly more efficient (no copy)
""")


# ============================================================================
# 4. CUSTOM SORTING
# ============================================================================

print("\n" + "=" * 70)
print("4. CUSTOM SORTING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Sorting with key Parameter
# ----------------------------------------------------------------------------
print("\n--- 4.1 Sorting with key Parameter ---")

# Sort by length
words = ["apple", "pie", "banana", "cat"]
sorted_by_length = sorted(words, key=len)
print(f"Words: {words}")
print(f"Sorted by length: {sorted_by_length}")

# Sort by absolute value
numbers = [-5, 2, -8, 1, -3]
sorted_abs = sorted(numbers, key=abs)
print(f"\nNumbers: {numbers}")
print(f"Sorted by absolute value: {sorted_abs}")

# Sort by second character
words = ["apple", "banana", "cherry"]
sorted_by_second = sorted(words, key=lambda x: x[1] if len(x) > 1 else '')
print(f"\nWords: {words}")
print(f"Sorted by second character: {sorted_by_second}")


# ----------------------------------------------------------------------------
# 4.2 Sorting Lists of Tuples/Dictionaries
# ----------------------------------------------------------------------------
print("\n--- 4.2 Sorting Lists of Tuples/Dictionaries ---")

# Sort list of tuples
students = [("Alice", 25), ("Bob", 20), ("Charlie", 22)]
sorted_by_age = sorted(students, key=lambda x: x[1])
print(f"Students: {students}")
print(f"Sorted by age: {sorted_by_age}")

# Sort by name (first element)
sorted_by_name = sorted(students, key=lambda x: x[0])
print(f"Sorted by name: {sorted_by_name}")

# Sort list of dictionaries
people = [
    {"name": "Alice", "age": 25, "city": "NY"},
    {"name": "Bob", "age": 20, "city": "LA"},
    {"name": "Charlie", "age": 22, "city": "NY"}
]
sorted_by_age = sorted(people, key=lambda x: x["age"])
print(f"\nPeople: {people}")
print(f"Sorted by age: {sorted_by_age}")


# ----------------------------------------------------------------------------
# 4.3 Sorting with Multiple Keys
# ----------------------------------------------------------------------------
print("\n--- 4.3 Sorting with Multiple Keys ---")

# Sort by multiple criteria
students = [
    ("Alice", 25, "A"),
    ("Bob", 25, "B"),
    ("Charlie", 20, "A"),
    ("David", 20, "B")
]

# Sort by age (ascending), then by grade (ascending)
sorted_multi = sorted(students, key=lambda x: (x[1], x[2]))
print(f"Students: {students}")
print(f"Sorted by age, then grade: {sorted_multi}")

# Sort by age (ascending), then by grade (descending)
sorted_multi_reverse = sorted(students, key=lambda x: (x[1], -ord(x[2])))
print(f"Sorted by age (asc), grade (desc): {sorted_multi_reverse}")


# ----------------------------------------------------------------------------
# 4.4 Using operator.itemgetter and operator.attrgetter
# ----------------------------------------------------------------------------
print("\n--- 4.4 Using operator.itemgetter ---")

from operator import itemgetter, attrgetter

# Sort list of tuples using itemgetter
students = [("Alice", 25), ("Bob", 20), ("Charlie", 22)]
sorted_by_age = sorted(students, key=itemgetter(1))
print(f"Students: {students}")
print(f"Sorted by age (itemgetter): {sorted_by_age}")

# Sort by multiple indices
students = [
    ("Alice", 25, "A"),
    ("Bob", 25, "B"),
    ("Charlie", 20, "A")
]
sorted_multi = sorted(students, key=itemgetter(1, 2))
print(f"\nSorted by age, then grade: {sorted_multi}")


# ----------------------------------------------------------------------------
# 4.5 Complex Sorting Examples
# ----------------------------------------------------------------------------
print("\n--- 4.5 Complex Sorting Examples ---")

# Sort strings by length, then alphabetically
words = ["apple", "pie", "banana", "cat", "date"]
sorted_complex = sorted(words, key=lambda x: (len(x), x))
print(f"Words: {words}")
print(f"Sorted by length, then alphabetically: {sorted_complex}")

# Sort numbers by sum of digits
def sum_of_digits(n):
    return sum(int(d) for d in str(abs(n)))

numbers = [123, 45, 67, 890, 12]
sorted_by_digit_sum = sorted(numbers, key=sum_of_digits)
print(f"\nNumbers: {numbers}")
print(f"Sorted by sum of digits: {sorted_by_digit_sum}")

# Sort by custom function
def custom_sort_key(x):
    """Sort by: negative of value if even, value if odd."""
    return (x % 2, -x if x % 2 == 0 else x)

numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_custom = sorted(numbers, key=custom_sort_key)
print(f"\nNumbers: {numbers}")
print(f"Custom sorted: {sorted_custom}")


# ----------------------------------------------------------------------------
# 4.6 Sorting with Case-Insensitive Comparison
# ----------------------------------------------------------------------------
print("\n--- 4.6 Sorting with Case-Insensitive Comparison ---")

words = ["Apple", "banana", "Cherry", "date"]
sorted_case_insensitive = sorted(words, key=str.lower)
print(f"Words: {words}")
print(f"Case-insensitive sorted: {sorted_case_insensitive}")


# ============================================================================
# 5. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICAL EXAMPLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Finding Element in Sorted Array
# ----------------------------------------------------------------------------
print("\n--- 5.1 Finding Element in Sorted Array ---")

def find_in_sorted(arr: List[int], target: int) -> bool:
    """Check if target exists in sorted array."""
    return binary_search(arr, target) != -1

arr = [1, 3, 5, 7, 9, 11, 13]
target = 7
result = find_in_sorted(arr, target)
print(f"Array: {arr}, Target: {target}")
print(f"Found: {result}")


# ----------------------------------------------------------------------------
# 5.2 Sorting and Searching Workflow
# ----------------------------------------------------------------------------
print("\n--- 5.2 Sorting and Searching Workflow ---")

# Unsorted data
data = [5, 2, 8, 1, 9, 3, 7, 4, 6]
print(f"Unsorted: {data}")

# Sort first
sorted_data = sorted(data)
print(f"Sorted: {sorted_data}")

# Now can use binary search
target = 7
index = binary_search(sorted_data, target)
print(f"Found {target} at index: {index}")


# ----------------------------------------------------------------------------
# 5.3 Custom Sorting for Complex Data
# ----------------------------------------------------------------------------
print("\n--- 5.3 Custom Sorting for Complex Data ---")

# Sort products by price, then by name
products = [
    {"name": "Laptop", "price": 1000, "category": "Electronics"},
    {"name": "Phone", "price": 500, "category": "Electronics"},
    {"name": "Tablet", "price": 500, "category": "Electronics"},
    {"name": "Book", "price": 20, "category": "Books"}
]

sorted_products = sorted(products, key=lambda x: (x["price"], x["name"]))
print("Products sorted by price, then name:")
for p in sorted_products:
    print(f"  {p}")


# ============================================================================
# 6. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Linear search for maximum
print("\n--- Exercise 1: Find Maximum ---")
def find_maximum(arr: List[int]) -> int:
    """Find maximum using linear search."""
    if not arr:
        return None
    max_val = arr[0]
    for val in arr[1:]:
        if val > max_val:
            max_val = val
    return max_val

arr = [5, 2, 8, 1, 9, 3]
result = find_maximum(arr)
print(f"Array: {arr}, Maximum: {result}")

# Exercise 2: Binary search for range
print("\n--- Exercise 2: Find Range ---")
def find_range(arr: List[int], target: int) -> tuple:
    """Find first and last occurrence of target."""
    first = binary_search_first_occurrence(arr, target)
    last = binary_search_last_occurrence(arr, target)
    return (first, last)

arr = [1, 2, 2, 2, 3, 4, 5]
target = 2
result = find_range(arr, target)
print(f"Array: {arr}, Target: {target}")
print(f"Range: {result}")

# Exercise 3: Sort by multiple criteria
print("\n--- Exercise 3: Sort by Multiple Criteria ---")
students = [
    ("Alice", 25, 85),
    ("Bob", 20, 90),
    ("Charlie", 25, 85),
    ("David", 20, 80)
]
# Sort by age (desc), then by score (desc)
sorted_students = sorted(students, key=lambda x: (-x[1], -x[2]))
print(f"Students: {students}")
print(f"Sorted by age (desc), score (desc): {sorted_students}")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. LINEAR SEARCH:
   - Time: O(n), Space: O(1)
   - Works on any array (sorted or unsorted)
   - Simple to implement
   - Use when: Unsorted data, small dataset, or need all occurrences

2. BINARY SEARCH:
   - Time: O(log n), Space: O(1) iterative, O(log n) recursive
   - Requires sorted array
   - Much faster than linear for large datasets
   - Use when: Sorted data, need to search multiple times
   - Variations: First/last occurrence, insertion position

3. BUILT-IN SORTING:
   - sorted(): Returns new list, works on any iterable
   - list.sort(): Modifies in-place, returns None
   - Time: O(n log n) average case
   - Use sorted() when you need original unchanged
   - Use sort() when you can modify original

4. CUSTOM SORTING:
   - Use key parameter for custom comparison
   - key can be function, lambda, or operator function
   - Multiple keys: key=lambda x: (x[1], x[2])
   - Reverse order: reverse=True or negate key
   - Use itemgetter/attrgetter for cleaner code

5. COMMON PATTERNS:
   - Sort by length: key=len
   - Sort by attribute: key=lambda x: x.attr
   - Sort by multiple: key=lambda x: (x[1], x[2])
   - Case-insensitive: key=str.lower
   - Custom function: key=my_function

6. BEST PRACTICES:
   - Use binary search only on sorted arrays
   - Prefer sorted() if you need original unchanged
   - Use key parameter for complex sorting
   - Consider time complexity (O(n) vs O(log n))
   - Handle edge cases (empty array, not found)

7. INTERVIEW TIPS:
   - Know when to use linear vs binary search
   - Understand sorted() vs sort() difference
   - Be comfortable with lambda functions
   - Know how to sort by multiple criteria
   - Practice binary search variations
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Searching & Sorting Guide Ready!")
    print("=" * 70)
