"""
Python Sets - Interview Preparation
Topic 2.4: Sets

This module covers:
- Creation: Set literals, set(), set comprehensions
- Operations: Union, intersection, difference, symmetric_difference
- Methods: add, remove, discard, pop, clear, update
- Set Comprehensions: Basic usage
"""

# ============================================================================
# 1. SET CREATION
# ============================================================================

print("=" * 70)
print("1. SET CREATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Set Literals
# ----------------------------------------------------------------------------
print("\n--- Set Literals ---")

# Empty set (note: {} creates empty dict, not set!)
empty_set = set()
print(f"Empty set: {empty_set}")
print(f"Type: {type(empty_set)}")

# Set with elements
numbers = {1, 2, 3, 4, 5}
print(f"Numbers set: {numbers}")

# Sets automatically remove duplicates
duplicates = {1, 2, 2, 3, 3, 3}
print(f"Set with duplicates: {duplicates}")  # {1, 2, 3}

# Sets are unordered (Python 3.7+ maintains insertion order, but don't rely on it)
mixed = {"apple", "banana", "cherry"}
print(f"Mixed set: {mixed}")

# Sets can contain only hashable (immutable) elements
valid_set = {1, 2, 3, "hello", (4, 5)}
print(f"Valid set: {valid_set}")

# Sets cannot contain mutable elements
# invalid_set = {1, 2, [3, 4]}  # TypeError: unhashable type: 'list'
# invalid_set = {1, 2, {"a": 1}}  # TypeError: unhashable type: 'dict'


# ----------------------------------------------------------------------------
# 1.2 Using set() Constructor
# ----------------------------------------------------------------------------
print("\n--- Using set() Constructor ---")

# From list
list_data = [1, 2, 3, 2, 3, 4]
set_from_list = set(list_data)
print(f"set([1, 2, 3, 2, 3, 4]): {set_from_list}")

# From string (iterable)
chars = set("hello")
print(f"set('hello'): {chars}")  # {'h', 'e', 'l', 'o'} - duplicates removed

# From tuple
tuple_data = (1, 2, 3, 2, 3)
set_from_tuple = set(tuple_data)
print(f"set((1, 2, 3, 2, 3)): {set_from_tuple}")

# From range
numbers = set(range(5))
print(f"set(range(5)): {numbers}")

# From another set (creates new set)
original = {1, 2, 3}
copied = set(original)
print(f"Original: {original}, Copied: {copied}")
print(f"Same object? {original is copied}")  # False


# ----------------------------------------------------------------------------
# 1.3 Set Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Set Comprehensions ---")

# Basic set comprehension
# Syntax: {expression for item in iterable}
squares = {x ** 2 for x in range(5)}
print(f"Squares: {squares}")

# With condition
evens = {x for x in range(10) if x % 2 == 0}
print(f"Even numbers: {evens}")

# From string
unique_chars = {char.upper() for char in "hello"}
print(f"Unique uppercase chars: {unique_chars}")

# More examples in section 4


# ----------------------------------------------------------------------------
# 1.4 Set Creation Methods Comparison
# ----------------------------------------------------------------------------
print("\n--- Set Creation Methods Comparison ---")

# Method 1: Literal
set1 = {1, 2, 3}

# Method 2: set() constructor
set2 = set([1, 2, 3])

# Method 3: Set comprehension
set3 = {x for x in [1, 2, 3]}

print(f"All methods create same set: {set1 == set2 == set3}")


# ----------------------------------------------------------------------------
# 1.5 Important: {} vs set()
# ----------------------------------------------------------------------------
print("\n--- Important: {} vs set() ---")

# {} creates empty dictionary, not set!
empty_dict = {}
print(f"{{}} creates: {type(empty_dict)}")  # <class 'dict'>

# Use set() for empty set
empty_set = set()
print(f"set() creates: {type(empty_set)}")  # <class 'set'>

# {} with elements creates set
non_empty_set = {1, 2, 3}
print(f"{{1, 2, 3}} creates: {type(non_empty_set)}")  # <class 'set'>


# ============================================================================
# 2. SET OPERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("2. SET OPERATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Union
# ----------------------------------------------------------------------------
print("\n--- Union ---")

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Method 1: union() method
union1 = set1.union(set2)
print(f"{set1} ∪ {set2} = {union1}")

# Method 2: | operator
union2 = set1 | set2
print(f"{set1} | {set2} = {union2}")

# Union with multiple sets
set3 = {5, 6, 7, 8}
union3 = set1.union(set2, set3)
print(f"Union of three sets: {union3}")

# Update in-place with |=
set1_copy = set1.copy()
set1_copy |= set2
print(f"After |=: {set1_copy}")


# ----------------------------------------------------------------------------
# 2.2 Intersection
# ----------------------------------------------------------------------------
print("\n--- Intersection ---")

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Method 1: intersection() method
intersection1 = set1.intersection(set2)
print(f"{set1} ∩ {set2} = {intersection1}")

# Method 2: & operator
intersection2 = set1 & set2
print(f"{set1} & {set2} = {intersection2}")

# Intersection with multiple sets
set3 = {4, 5, 6, 7}
intersection3 = set1.intersection(set2, set3)
print(f"Intersection of three sets: {intersection3}")

# Update in-place with &=
set1_copy = set1.copy()
set1_copy &= set2
print(f"After &=: {set1_copy}")


# ----------------------------------------------------------------------------
# 2.3 Difference
# ----------------------------------------------------------------------------
print("\n--- Difference ---")

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Method 1: difference() method
difference1 = set1.difference(set2)
print(f"{set1} - {set2} = {difference1}")  # Elements in set1 but not in set2

# Method 2: - operator
difference2 = set1 - set2
print(f"{set1} - {set2} = {difference2}")

# Difference is not symmetric
difference_reverse = set2 - set1
print(f"{set2} - {set1} = {difference_reverse}")

# Difference with multiple sets
set3 = {4, 5}
difference3 = set1.difference(set2, set3)
print(f"Difference with multiple sets: {difference3}")

# Update in-place with -=
set1_copy = set1.copy()
set1_copy -= set2
print(f"After -=: {set1_copy}")


# ----------------------------------------------------------------------------
# 2.4 Symmetric Difference
# ----------------------------------------------------------------------------
print("\n--- Symmetric Difference ---")

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Method 1: symmetric_difference() method
sym_diff1 = set1.symmetric_difference(set2)
print(f"{set1} △ {set2} = {sym_diff1}")  # Elements in either set, but not both

# Method 2: ^ operator
sym_diff2 = set1 ^ set2
print(f"{set1} ^ {set2} = {sym_diff2}")

# Symmetric difference is symmetric
sym_diff_reverse = set2 ^ set1
print(f"{set2} ^ {set1} = {sym_diff_reverse}")
print(f"Symmetric: {sym_diff1 == sym_diff_reverse}")

# Update in-place with ^=
set1_copy = set1.copy()
set1_copy ^= set2
print(f"After ^=: {set1_copy}")


# ----------------------------------------------------------------------------
# 2.5 Set Comparison Operations
# ----------------------------------------------------------------------------
print("\n--- Set Comparison Operations ---")

set1 = {1, 2, 3}
set2 = {1, 2, 3, 4}
set3 = {1, 2, 3}

# Subset
print(f"{set1} ⊆ {set2}: {set1.issubset(set2)}")
print(f"{set1} <= {set2}: {set1 <= set2}")  # Same as issubset
print(f"{set1} < {set2}: {set1 < set2}")  # Proper subset

# Superset
print(f"{set2} ⊇ {set1}: {set2.issuperset(set1)}")
print(f"{set2} >= {set1}: {set2 >= set1}")  # Same as issuperset
print(f"{set2} > {set1}: {set2 > set1}")  # Proper superset

# Disjoint (no common elements)
set4 = {5, 6, 7}
print(f"{set1} and {set4} are disjoint: {set1.isdisjoint(set4)}")
print(f"{set1} and {set2} are disjoint: {set1.isdisjoint(set2)}")


# ----------------------------------------------------------------------------
# 2.6 Membership Testing
# ----------------------------------------------------------------------------
print("\n--- Membership Testing ---")

numbers = {1, 2, 3, 4, 5}

# Check membership
print(f"3 in {numbers}: {3 in numbers}")
print(f"6 in {numbers}: {6 in numbers}")
print(f"3 not in {numbers}: {3 not in numbers}")

# Membership testing is O(1) for sets (very fast)
# Much faster than checking membership in lists


# ============================================================================
# 3. SET METHODS
# ============================================================================

print("\n" + "=" * 70)
print("3. SET METHODS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 add() - Add Single Element
# ----------------------------------------------------------------------------
print("\n--- add() - Add Single Element ---")

numbers = {1, 2, 3}
print(f"Original: {numbers}")

numbers.add(4)
print(f"After add(4): {numbers}")

numbers.add(2)  # Adding duplicate has no effect
print(f"After add(2): {numbers}")

# add() only adds one element
# numbers.add([5, 6])  # TypeError: unhashable type: 'list'


# ----------------------------------------------------------------------------
# 3.2 remove() - Remove Element
# ----------------------------------------------------------------------------
print("\n--- remove() - Remove Element ---")

numbers = {1, 2, 3, 4, 5}
print(f"Original: {numbers}")

numbers.remove(3)
print(f"After remove(3): {numbers}")

# KeyError if element doesn't exist
# numbers.remove(99)  # KeyError: 99

# Safe removal pattern
if 99 in numbers:
    numbers.remove(99)


# ----------------------------------------------------------------------------
# 3.3 discard() - Remove Element (Safe)
# ----------------------------------------------------------------------------
print("\n--- discard() - Remove Element (Safe) ---")

numbers = {1, 2, 3, 4, 5}
print(f"Original: {numbers}")

numbers.discard(3)
print(f"After discard(3): {numbers}")

# discard() doesn't raise error if element doesn't exist
numbers.discard(99)  # No error
print(f"After discard(99): {numbers}")

# Use discard() when you're not sure if element exists
# Use remove() when you want to know if element was present


# ----------------------------------------------------------------------------
# 3.4 pop() - Remove and Return Arbitrary Element
# ----------------------------------------------------------------------------
print("\n--- pop() - Remove and Return Arbitrary Element ---")

numbers = {1, 2, 3, 4, 5}
print(f"Original: {numbers}")

# pop() removes and returns arbitrary element
popped = numbers.pop()
print(f"Popped: {popped}, Remaining: {numbers}")

# pop() raises KeyError if set is empty
# empty = set()
# empty.pop()  # KeyError: 'pop from an empty set'

# pop() is useful for processing elements one by one
numbers = {1, 2, 3}
while numbers:
    element = numbers.pop()
    print(f"Processing: {element}")


# ----------------------------------------------------------------------------
# 3.5 clear() - Remove All Elements
# ----------------------------------------------------------------------------
print("\n--- clear() - Remove All Elements ---")

numbers = {1, 2, 3, 4, 5}
print(f"Original: {numbers}")

numbers.clear()
print(f"After clear(): {numbers}")


# ----------------------------------------------------------------------------
# 3.6 update() - Add Multiple Elements
# ----------------------------------------------------------------------------
print("\n--- update() - Add Multiple Elements ---")

numbers = {1, 2, 3}
print(f"Original: {numbers}")

# update() with another set
numbers.update({4, 5, 6})
print(f"After update({{4, 5, 6}}): {numbers}")

# update() with list
numbers.update([7, 8, 9])
print(f"After update([7, 8, 9]): {numbers}")

# update() with any iterable
numbers.update(range(10, 13))
print(f"After update(range(10, 13)): {numbers}")

# update() is equivalent to |=
numbers_copy = {1, 2, 3}
numbers_copy |= {4, 5, 6}
print(f"After |=: {numbers_copy}")


# ----------------------------------------------------------------------------
# 3.7 Other Set Methods
# ----------------------------------------------------------------------------
print("\n--- Other Set Methods ---")

numbers = {1, 2, 3, 4, 5}

# len() - Get number of elements
print(f"Length: {len(numbers)}")

# copy() - Shallow copy
copied = numbers.copy()
print(f"Copied: {copied}")
print(f"Same object? {numbers is copied}")  # False

# in operator - Membership testing
print(f"3 in numbers: {3 in numbers}")

# Iteration
print("Iterating set:")
for num in numbers:
    print(f"  {num}", end=" ")
print()


# ============================================================================
# 4. SET COMPREHENSIONS
# ============================================================================

print("\n" + "=" * 70)
print("4. SET COMPREHENSIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Basic Set Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Basic Set Comprehensions ---")

# Syntax: {expression for item in iterable}

# Square numbers
squares = {x ** 2 for x in range(5)}
print(f"Squares: {squares}")

# From string (removes duplicates automatically)
unique_chars = {char for char in "hello"}
print(f"Unique chars in 'hello': {unique_chars}")

# Transform elements
numbers = [1, 2, 3, 4, 5]
doubled = {x * 2 for x in numbers}
print(f"Doubled: {doubled}")


# ----------------------------------------------------------------------------
# 4.2 Set Comprehensions with Conditions
# ----------------------------------------------------------------------------
print("\n--- Set Comprehensions with Conditions ---")

# Syntax: {expression for item in iterable if condition}

# Even squares
even_squares = {x ** 2 for x in range(10) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# Filter and transform
words = ["apple", "banana", "cherry", "date"]
long_words = {word.upper() for word in words if len(word) > 5}
print(f"Long words (uppercase): {long_words}")


# ----------------------------------------------------------------------------
# 4.3 Set Comprehensions vs List Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Set Comprehensions vs List Comprehensions ---")

# List comprehension (preserves order, allows duplicates)
numbers = [1, 2, 2, 3, 3, 3]
squares_list = [x ** 2 for x in numbers]
print(f"List comprehension: {squares_list}")

# Set comprehension (removes duplicates, unordered)
squares_set = {x ** 2 for x in numbers}
print(f"Set comprehension: {squares_set}")


# ----------------------------------------------------------------------------
# 4.4 Common Set Comprehension Patterns
# ----------------------------------------------------------------------------
print("\n--- Common Set Comprehension Patterns ---")

# Remove duplicates from list
numbers = [1, 2, 2, 3, 3, 3, 4, 5]
unique = {x for x in numbers}
print(f"Unique numbers: {unique}")

# Extract unique characters
text = "hello world"
unique_chars = {char for char in text if char != ' '}
print(f"Unique chars: {unique_chars}")

# Filter and transform
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = {x for x in data if x % 2 == 0}
print(f"Even numbers: {evens}")


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Find unique elements
print("\n--- Exercise 1: Find Unique Elements ---")
def find_unique(items):
    """Find unique elements in list."""
    return set(items)

numbers = [1, 2, 2, 3, 3, 3, 4, 5]
unique = find_unique(numbers)
print(f"Unique elements in {numbers}: {unique}")


# Exercise 2: Check if two lists have common elements
print("\n--- Exercise 2: Check Common Elements ---")
def have_common(list1, list2):
    """Check if two lists have common elements."""
    return bool(set(list1) & set(list2))

list1 = [1, 2, 3, 4]
list2 = [3, 4, 5, 6]
list3 = [7, 8, 9]
print(f"{list1} and {list2} have common: {have_common(list1, list2)}")
print(f"{list1} and {list3} have common: {have_common(list1, list3)}")


# Exercise 3: Find elements in one set but not the other
print("\n--- Exercise 3: Set Difference ---")
def find_difference(set1, set2):
    """Find elements in set1 but not in set2."""
    return set1 - set2

set1 = {1, 2, 3, 4, 5}
set2 = {3, 4, 5, 6, 7}
diff = find_difference(set1, set2)
print(f"Elements in {set1} but not in {set2}: {diff}")


# Exercise 4: Find symmetric difference
print("\n--- Exercise 4: Symmetric Difference ---")
def find_symmetric_difference(set1, set2):
    """Find elements in either set but not both."""
    return set1 ^ set2

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
sym_diff = find_symmetric_difference(set1, set2)
print(f"Symmetric difference: {sym_diff}")


# Exercise 5: Check if one set is subset of another
print("\n--- Exercise 5: Subset Check ---")
def is_subset(set1, set2):
    """Check if set1 is subset of set2."""
    return set1 <= set2

set1 = {1, 2, 3}
set2 = {1, 2, 3, 4, 5}
set3 = {6, 7, 8}
print(f"{set1} ⊆ {set2}: {is_subset(set1, set2)}")
print(f"{set1} ⊆ {set3}: {is_subset(set1, set3)}")


# Exercise 6: Remove duplicates while preserving order (Python 3.7+)
print("\n--- Exercise 6: Remove Duplicates Preserving Order ---")
def remove_duplicates_ordered(items):
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

numbers = [1, 2, 2, 3, 3, 3, 4, 5]
unique_ordered = remove_duplicates_ordered(numbers)
print(f"Unique (ordered): {unique_ordered}")


# Exercise 7: Find all unique combinations
print("\n--- Exercise 7: Find All Unique Combinations ---")
def find_common_elements(*lists):
    """Find elements common to all lists."""
    if not lists:
        return set()
    result = set(lists[0])
    for lst in lists[1:]:
        result &= set(lst)
    return result

list1 = [1, 2, 3, 4]
list2 = [2, 3, 4, 5]
list3 = [3, 4, 5, 6]
common = find_common_elements(list1, list2, list3)
print(f"Common elements: {common}")


# Exercise 8: Set operations on multiple sets
print("\n--- Exercise 8: Multiple Set Operations ---")
def set_operations(set1, set2, set3):
    """Perform various set operations."""
    return {
        "union": set1 | set2 | set3,
        "intersection": set1 & set2 & set3,
        "all_in_set1": set1 - set2 - set3,
        "only_in_one": (set1 ^ set2 ^ set3) - (set1 & set2 & set3)
    }

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
set3 = {5, 6, 7, 8}
results = set_operations(set1, set2, set3)
print(f"Set operations results: {results}")


# ============================================================================
# 6. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between sets and lists?
print("\n--- Q1: Sets vs Lists ---")
print("""
Sets:
- Unordered (Python 3.7+ maintains insertion order, but don't rely on it)
- No duplicates
- Mutable
- Only hashable elements
- O(1) membership testing
- Use for: unique elements, membership testing, set operations

Lists:
- Ordered
- Allows duplicates
- Mutable
- Any elements
- O(n) membership testing
- Use for: ordered sequences, indexing, slicing
""")


# Q2: What's the difference between remove() and discard()?
print("\n--- Q2: remove() vs discard() ---")
print("""
remove(element):
- Raises KeyError if element doesn't exist
- Use when you want to know if element was present

discard(element):
- Doesn't raise error if element doesn't exist
- Use when you don't care if element was present
- Safer for general use
""")


# Q3: How to create an empty set?
print("\n--- Q3: Empty Set Creation ---")
print("""
Use set(), not {}
{} creates empty dictionary, not set
set() creates empty set
""")


# Q4: What are set operations?
print("\n--- Q4: Set Operations ---")
print("""
Union (|): Elements in either set
Intersection (&): Elements in both sets
Difference (-): Elements in first set but not second
Symmetric Difference (^): Elements in either set but not both
Subset (<=): All elements of first set in second
Superset (>=): All elements of second set in first
""")


# Q5: Why are sets faster for membership testing?
print("\n--- Q5: Set Membership Performance ---")
print("""
Sets use hash tables internally
Membership testing is O(1) average case
Lists use linear search, O(n) worst case
Use sets when you need fast membership testing
""")


# Q6: Can sets contain mutable elements?
print("\n--- Q6: Mutable Elements in Sets ---")
print("""
No, sets can only contain hashable (immutable) elements
Hashable: int, float, str, tuple (if elements hashable)
Not hashable: list, dict, set
Sets themselves are mutable but cannot contain mutable elements
""")


# Q7: What's the difference between set comprehension and list comprehension?
print("\n--- Q7: Set vs List Comprehensions ---")
print("""
Set comprehension: {expr for item in iterable}
- Removes duplicates automatically
- Unordered result
- Use when you need unique elements

List comprehension: [expr for item in iterable]
- Preserves duplicates
- Ordered result
- Use when order matters
""")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. SET CREATION:
   - Literal: {1, 2, 3} (not {} - that's empty dict)
   - set() constructor: set([1, 2, 3])
   - Set comprehensions: {x for x in iterable}
   - Empty set: set() (not {})

2. SET PROPERTIES:
   - Unordered (Python 3.7+ maintains insertion order)
   - No duplicates (automatically removed)
   - Mutable
   - Only hashable (immutable) elements
   - Fast membership testing O(1)

3. SET OPERATIONS:
   - Union: | or union()
   - Intersection: & or intersection()
   - Difference: - or difference()
   - Symmetric Difference: ^ or symmetric_difference()
   - Subset: <= or issubset()
   - Superset: >= or issuperset()
   - Disjoint: isdisjoint()

4. SET METHODS:
   - add(element): Add single element
   - remove(element): Remove element (raises KeyError if missing)
   - discard(element): Remove element (no error if missing)
   - pop(): Remove and return arbitrary element
   - clear(): Remove all elements
   - update(iterable): Add multiple elements
   - copy(): Shallow copy

5. SET COMPREHENSIONS:
   - Syntax: {expression for item in iterable if condition}
   - Automatically removes duplicates
   - More Pythonic than loops for set creation

6. COMMON USE CASES:
   - Remove duplicates from list
   - Fast membership testing
   - Set operations (union, intersection, etc.)
   - Finding unique elements
   - Checking for common elements

7. PERFORMANCE:
   - Membership testing: O(1) average
   - Adding/removing: O(1) average
   - Much faster than lists for membership testing
   - Use sets when you need fast lookups

8. BEST PRACTICES:
   - Use set() for empty set (not {})
   - Use discard() for safe removal
   - Use sets for membership testing
   - Use set comprehensions when appropriate
   - Remember sets are unordered (don't rely on order)
   - Sets can only contain hashable elements
""")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
