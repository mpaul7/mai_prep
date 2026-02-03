"""
Hash Tables & Sets - Interview Preparation
Topic 6.3: Hash Tables & Sets

This module covers:
- Frequency Counting: Using dictionaries/Counter
- Set Operations: Finding unique elements, intersections
- Lookup Optimization: O(1) lookups
"""

from collections import Counter, defaultdict
from typing import List, Set, Dict, Any

# ============================================================================
# 1. FREQUENCY COUNTING
# ============================================================================

print("=" * 70)
print("1. FREQUENCY COUNTING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Using Dictionaries for Frequency Counting
# ----------------------------------------------------------------------------
print("\n--- 1.1 Using Dictionaries for Frequency Counting ---")

def count_frequency_manual(items: List) -> Dict:
    """
    Count frequency manually using dictionary.
    Time: O(n), Space: O(k) where k is unique elements
    """
    freq = {}
    for item in items:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    return freq

# Example
items = ['a', 'b', 'a', 'c', 'b', 'a']
freq = count_frequency_manual(items)
print(f"Items: {items}")
print(f"Frequency: {freq}")


def count_frequency_get(items: List) -> Dict:
    """
    Count frequency using dict.get() method.
    More Pythonic than manual check.
    """
    freq = {}
    for item in items:
        freq[item] = freq.get(item, 0) + 1
    return freq

# Example
items = ['a', 'b', 'a', 'c', 'b', 'a']
freq = count_frequency_get(items)
print(f"\nItems: {items}")
print(f"Frequency (get method): {freq}")


def count_frequency_defaultdict(items: List) -> Dict:
    """
    Count frequency using defaultdict.
    Cleanest approach.
    """
    freq = defaultdict(int)
    for item in items:
        freq[item] += 1
    return dict(freq)

# Example
items = ['a', 'b', 'a', 'c', 'b', 'a']
freq = count_frequency_defaultdict(items)
print(f"\nItems: {items}")
print(f"Frequency (defaultdict): {freq}")


# ----------------------------------------------------------------------------
# 1.2 Using Counter for Frequency Counting
# ----------------------------------------------------------------------------
print("\n--- 1.2 Using Counter for Frequency Counting ---")

# Counter is specialized for frequency counting
items = ['a', 'b', 'a', 'c', 'b', 'a']
counter = Counter(items)
print(f"Items: {items}")
print(f"Counter: {counter}")

# Counter from string
text = "hello"
char_counter = Counter(text)
print(f"\nText: '{text}'")
print(f"Character counter: {char_counter}")

# Counter from dictionary
counter = Counter({'a': 3, 'b': 2, 'c': 1})
print(f"\nCounter from dict: {counter}")

# Accessing counts
print(f"\nCount of 'a': {counter['a']}")
print(f"Count of 'z' (missing): {counter['z']}")  # Returns 0, no KeyError


# ----------------------------------------------------------------------------
# 1.3 Counter Methods
# ----------------------------------------------------------------------------
print("\n--- 1.3 Counter Methods ---")

counter = Counter(['apple', 'banana', 'apple', 'cherry', 'banana', 'apple'])
print(f"Counter: {counter}")

# most_common() - Get most common elements
print(f"\nMost common 2: {counter.most_common(2)}")
print(f"Most common (all): {counter.most_common()}")

# elements() - Get all elements as iterator
elements = list(counter.elements())
print(f"\nElements: {elements}")

# update() - Add counts
counter.update(['apple', 'date'])
print(f"\nAfter update: {counter}")

# subtract() - Subtract counts
counter.subtract(['apple', 'banana'])
print(f"After subtract: {counter}")


# ----------------------------------------------------------------------------
# 1.4 Counter Operations
# ----------------------------------------------------------------------------
print("\n--- 1.4 Counter Operations ---")

counter1 = Counter(['a', 'b', 'c', 'a'])
counter2 = Counter(['a', 'b', 'b'])

print(f"Counter1: {counter1}")
print(f"Counter2: {counter2}")

# Addition (combine counts)
combined = counter1 + counter2
print(f"\nCounter1 + Counter2: {combined}")

# Subtraction (only positive counts)
subtracted = counter1 - counter2
print(f"Counter1 - Counter2: {subtracted}")

# Intersection (minimum counts)
intersection = counter1 & counter2
print(f"Counter1 & Counter2: {intersection}")

# Union (maximum counts)
union = counter1 | counter2
print(f"Counter1 | Counter2: {union}")


# ----------------------------------------------------------------------------
# 1.5 Common Frequency Counting Patterns
# ----------------------------------------------------------------------------
print("\n--- 1.5 Common Frequency Counting Patterns ---")

# Count character frequencies
text = "hello world"
char_count = Counter(text)
print(f"Text: '{text}'")
print(f"Character frequencies: {char_count}")

# Count word frequencies
words = "the quick brown fox jumps over the lazy dog".split()
word_count = Counter(words)
print(f"\nWords: {words}")
print(f"Word frequencies: {word_count}")

# Find most common elements
numbers = [1, 2, 3, 2, 3, 3, 4, 4, 4, 4]
num_count = Counter(numbers)
print(f"\nNumbers: {numbers}")
print(f"Most common 2: {num_count.most_common(2)}")

# Top K frequent elements
def top_k_frequent(items: List, k: int) -> List:
    """Find top k frequent elements."""
    counter = Counter(items)
    return [item for item, count in counter.most_common(k)]

items = ['a', 'b', 'a', 'c', 'b', 'a', 'd']
result = top_k_frequent(items, 2)
print(f"\nItems: {items}")
print(f"Top 2 frequent: {result}")


# ============================================================================
# 2. SET OPERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("2. SET OPERATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Finding Unique Elements
# ----------------------------------------------------------------------------
print("\n--- 2.1 Finding Unique Elements ---")

# Convert list to set to get unique elements
items = [1, 2, 3, 2, 3, 4, 5, 1]
unique = set(items)
print(f"Items: {items}")
print(f"Unique elements: {unique}")

# Convert back to list if needed
unique_list = list(set(items))
print(f"Unique as list: {unique_list}")

# Preserve order (Python 3.7+ dicts maintain insertion order)
def unique_preserve_order(items: List) -> List:
    """Get unique elements preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

items = [1, 2, 3, 2, 3, 4, 5, 1]
unique_ordered = unique_preserve_order(items)
print(f"\nItems: {items}")
print(f"Unique (preserving order): {unique_ordered}")


# ----------------------------------------------------------------------------
# 2.2 Set Intersection
# ----------------------------------------------------------------------------
print("\n--- 2.2 Set Intersection ---")

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

print(f"Set1: {set1}")
print(f"Set2: {set2}")

# Intersection - elements in both sets
intersection = set1 & set2
print(f"\nIntersection (set1 & set2): {intersection}")

# Using intersection() method
intersection_method = set1.intersection(set2)
print(f"Intersection (method): {intersection_method}")

# Multiple sets
set3 = {5, 6, 9}
intersection_all = set1 & set2 & set3
print(f"\nSet3: {set3}")
print(f"Intersection of all: {intersection_all}")


# ----------------------------------------------------------------------------
# 2.3 Set Union
# ----------------------------------------------------------------------------
print("\n--- 2.3 Set Union ---")

set1 = {1, 2, 3}
set2 = {3, 4, 5}

print(f"Set1: {set1}")
print(f"Set2: {set2}")

# Union - all unique elements from both sets
union = set1 | set2
print(f"\nUnion (set1 | set2): {union}")

# Using union() method
union_method = set1.union(set2)
print(f"Union (method): {union_method}")


# ----------------------------------------------------------------------------
# 2.4 Set Difference
# ----------------------------------------------------------------------------
print("\n--- 2.4 Set Difference ---")

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7}

print(f"Set1: {set1}")
print(f"Set2: {set2}")

# Difference - elements in set1 but not in set2
difference = set1 - set2
print(f"\nDifference (set1 - set2): {difference}")

# Symmetric difference - elements in either set but not both
symmetric_diff = set1 ^ set2
print(f"Symmetric difference (set1 ^ set2): {symmetric_diff}")


# ----------------------------------------------------------------------------
# 2.5 Set Membership and Subset Operations
# ----------------------------------------------------------------------------
print("\n--- 2.5 Set Membership and Subset Operations ---")

set1 = {1, 2, 3, 4, 5}
set2 = {2, 3, 4}

print(f"Set1: {set1}")
print(f"Set2: {set2}")

# Check if subset
is_subset = set2.issubset(set1)
print(f"\nset2 is subset of set1: {is_subset}")
print(f"Using <= operator: {set2 <= set1}")

# Check if superset
is_superset = set1.issuperset(set2)
print(f"\nset1 is superset of set2: {is_superset}")
print(f"Using >= operator: {set1 >= set2}")

# Check if disjoint (no common elements)
set3 = {6, 7, 8}
is_disjoint = set1.isdisjoint(set3)
print(f"\nset1 and set3 are disjoint: {is_disjoint}")


# ----------------------------------------------------------------------------
# 2.6 Common Set Operations Patterns
# ----------------------------------------------------------------------------
print("\n--- 2.6 Common Set Operations Patterns ---")

# Find common elements in two lists
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
common = set(list1) & set(list2)
print(f"List1: {list1}")
print(f"List2: {list2}")
print(f"Common elements: {common}")

# Find elements in list1 but not in list2
only_in_list1 = set(list1) - set(list2)
print(f"\nOnly in list1: {only_in_list1}")

# Find all unique elements from both lists
all_unique = set(list1) | set(list2)
print(f"All unique elements: {all_unique}")

# Check if two lists have any common elements
has_common = bool(set(list1) & set(list2))
print(f"\nHas common elements: {has_common}")


# ============================================================================
# 3. LOOKUP OPTIMIZATION
# ============================================================================

print("\n" + "=" * 70)
print("3. LOOKUP OPTIMIZATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 O(1) Lookups with Dictionaries
# ----------------------------------------------------------------------------
print("\n--- 3.1 O(1) Lookups with Dictionaries ---")

# Dictionary provides O(1) average case lookup
my_dict = {'apple': 1, 'banana': 2, 'cherry': 3}

# O(1) lookup
value = my_dict['apple']
print(f"Dictionary: {my_dict}")
print(f"Lookup 'apple': {value}")

# O(1) check if key exists
has_key = 'banana' in my_dict
print(f"'banana' in dict: {has_key}")

# O(1) get with default
value = my_dict.get('orange', 0)
print(f"get('orange', 0): {value}")


# ----------------------------------------------------------------------------
# 3.2 O(1) Lookups with Sets
# ----------------------------------------------------------------------------
print("\n--- 3.2 O(1) Lookups with Sets ---")

# Set provides O(1) average case membership test
my_set = {1, 2, 3, 4, 5}

# O(1) membership test
is_member = 3 in my_set
print(f"Set: {my_set}")
print(f"3 in set: {is_member}")

# O(1) add
my_set.add(6)
print(f"After add(6): {my_set}")

# O(1) remove
my_set.remove(6)
print(f"After remove(6): {my_set}")


# ----------------------------------------------------------------------------
# 3.3 Optimizing List Lookups
# ----------------------------------------------------------------------------
print("\n--- 3.3 Optimizing List Lookups ---")

# List membership test is O(n)
my_list = [1, 2, 3, 4, 5]
print(f"List: {my_list}")

# O(n) - slow for large lists
is_in_list = 3 in my_list
print(f"3 in list: {is_in_list}")

# Convert to set for O(1) lookups
my_set = set(my_list)
is_in_set = 3 in my_set
print(f"3 in set: {is_in_set}")

# When to convert: if checking multiple times
def check_multiple_items_list(items: List, targets: List) -> List[bool]:
    """Check multiple items - inefficient with list."""
    return [target in items for target in targets]

def check_multiple_items_set(items: List, targets: List) -> List[bool]:
    """Check multiple items - efficient with set."""
    items_set = set(items)
    return [target in items_set for target in targets]

items = list(range(1000))
targets = [100, 500, 999]
print(f"\nChecking {len(targets)} items in list of {len(items)}")
print(f"Using list: {check_multiple_items_list(items, targets)}")
print(f"Using set: {check_multiple_items_set(items, targets)}")


# ----------------------------------------------------------------------------
# 3.4 Creating Lookup Tables
# ----------------------------------------------------------------------------
print("\n--- 3.4 Creating Lookup Tables ---")

# Create index for fast lookups
students = [
    {'id': 1, 'name': 'Alice', 'age': 25},
    {'id': 2, 'name': 'Bob', 'age': 30},
    {'id': 3, 'name': 'Charlie', 'age': 25}
]

# Create lookup by id
id_lookup = {student['id']: student for student in students}
print(f"Students: {students}")
print(f"ID lookup: {id_lookup}")

# O(1) lookup by id
student = id_lookup[2]
print(f"\nStudent with id=2: {student}")

# Create lookup by name
name_lookup = {student['name']: student for student in students}
print(f"\nName lookup: {name_lookup}")

# O(1) lookup by name
student = name_lookup['Alice']
print(f"Student named 'Alice': {student}")


# ----------------------------------------------------------------------------
# 3.5 Reverse Lookup
# ----------------------------------------------------------------------------
print("\n--- 3.5 Reverse Lookup ---")

# Create reverse lookup (value -> key)
forward_dict = {'a': 1, 'b': 2, 'c': 3}
reverse_dict = {v: k for k, v in forward_dict.items()}
print(f"Forward dict: {forward_dict}")
print(f"Reverse dict: {reverse_dict}")

# O(1) reverse lookup
key = reverse_dict[2]
print(f"\nKey for value 2: {key}")

# Note: Only works if values are unique!


# ----------------------------------------------------------------------------
# 3.6 Caching with Dictionaries
# ----------------------------------------------------------------------------
print("\n--- 3.6 Caching with Dictionaries ---")

# Simple cache using dictionary
cache = {}

def expensive_function(n):
    """Expensive computation."""
    if n in cache:
        return cache[n]
    # Simulate expensive computation
    result = n ** 2
    cache[n] = result
    return result

print("First call (computes):", expensive_function(5))
print("Second call (from cache):", expensive_function(5))
print(f"Cache: {cache}")


# Using functools.lru_cache (built-in caching)
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_function(n):
    """Function with built-in caching."""
    return n ** 2

print(f"\nFirst call: {cached_function(5)}")
print(f"Second call: {cached_function(5)}")


# ============================================================================
# 4. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("4. PRACTICAL EXAMPLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Finding Duplicates
# ----------------------------------------------------------------------------
print("\n--- 4.1 Finding Duplicates ---")

def find_duplicates(items: List) -> List:
    """Find duplicate elements."""
    seen = set()
    duplicates = set()
    
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    
    return list(duplicates)

items = [1, 2, 3, 2, 4, 3, 5]
duplicates = find_duplicates(items)
print(f"Items: {items}")
print(f"Duplicates: {duplicates}")


# ----------------------------------------------------------------------------
# 4.2 Grouping Elements
# ----------------------------------------------------------------------------
print("\n--- 4.2 Grouping Elements ---")

def group_by_key(items: List[Dict], key: str) -> Dict:
    """Group items by key value."""
    groups = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    return dict(groups)

students = [
    {'name': 'Alice', 'department': 'IT'},
    {'name': 'Bob', 'department': 'Sales'},
    {'name': 'Charlie', 'department': 'IT'},
    {'name': 'David', 'department': 'Sales'}
]

grouped = group_by_key(students, 'department')
print(f"Students: {students}")
print(f"Grouped by department: {grouped}")


# ----------------------------------------------------------------------------
# 4.3 Finding Common Elements
# ----------------------------------------------------------------------------
print("\n--- 4.3 Finding Common Elements ---")

def find_common_elements(*lists):
    """Find elements common to all lists."""
    if not lists:
        return []
    
    common = set(lists[0])
    for lst in lists[1:]:
        common &= set(lst)
    
    return list(common)

list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7]
list3 = [5, 6, 7, 8]

common = find_common_elements(list1, list2, list3)
print(f"List1: {list1}")
print(f"List2: {list2}")
print(f"List3: {list3}")
print(f"Common to all: {common}")


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Count frequency
print("\n--- Exercise 1: Count Frequency ---")
def count_chars(text: str) -> Dict:
    """Count character frequency."""
    return dict(Counter(text))

text = "hello"
result = count_chars(text)
print(f"Text: '{text}'")
print(f"Character count: {result}")

# Exercise 2: Find intersection
print("\n--- Exercise 2: Find Intersection ---")
def find_intersection(list1: List, list2: List) -> List:
    """Find intersection of two lists."""
    return list(set(list1) & set(list2))

list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7]
result = find_intersection(list1, list2)
print(f"List1: {list1}, List2: {list2}")
print(f"Intersection: {result}")

# Exercise 3: Optimize lookup
print("\n--- Exercise 3: Optimize Lookup ---")
def check_items_efficient(items: List, targets: List) -> List[bool]:
    """Check if targets exist in items efficiently."""
    items_set = set(items)
    return [target in items_set for target in targets]

items = [1, 2, 3, 4, 5]
targets = [3, 6, 1]
result = check_items_efficient(items, targets)
print(f"Items: {items}, Targets: {targets}")
print(f"Results: {result}")


# ============================================================================
# 6. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("6. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. FREQUENCY COUNTING:
   - Use Counter for frequency counting (most Pythonic)
   - Use defaultdict(int) for manual counting
   - Use dict.get() as alternative
   - Counter.most_common(k) for top k elements
   - Time: O(n), Space: O(k) where k is unique elements

2. SET OPERATIONS:
   - Use sets for O(1) membership testing
   - Intersection: set1 & set2
   - Union: set1 | set2
   - Difference: set1 - set2
   - Symmetric difference: set1 ^ set2
   - Convert list to set for fast lookups

3. LOOKUP OPTIMIZATION:
   - Dictionary: O(1) average case lookup
   - Set: O(1) average case membership test
   - List: O(n) membership test (slow!)
   - Convert list to set if checking multiple times
   - Create lookup tables for fast access

4. WHEN TO USE EACH:
   - Counter: Frequency counting, top k elements
   - Set: Unique elements, membership testing, set operations
   - Dict: Key-value lookups, caching, indexing
   - List: Ordered data, when order matters

5. COMMON PATTERNS:
   - Count frequency: Counter(items)
   - Find unique: set(items)
   - Fast lookup: set(items) then 'item in set'
   - Group by: defaultdict(list)
   - Top k: Counter(items).most_common(k)

6. PERFORMANCE:
   - Dict/Set: O(1) average case operations
   - List: O(n) for membership test
   - Convert list to set for multiple lookups
   - Use Counter for counting (optimized)

7. BEST PRACTICES:
   - Use Counter instead of manual dict counting
   - Convert to set for membership testing
   - Create lookup tables for repeated access
   - Use defaultdict to avoid key checks
   - Know time complexity of operations

8. INTERVIEW TIPS:
   - Recognize when O(1) lookup is needed
   - Use set for membership testing
   - Use Counter for frequency problems
   - Convert list to set for optimization
   - Explain time complexity improvements
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Hash Tables & Sets Guide Ready!")
    print("=" * 70)
