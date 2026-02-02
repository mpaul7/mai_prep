"""
Python Tuples - Interview Preparation
Topic 2.2: Tuples

This module covers:
- Creation: Tuple literals, tuple()
- Immutability: When and why to use tuples
- Unpacking: Multiple assignment, tuple unpacking
- Methods: count, index
"""

# ============================================================================
# 1. TUPLE CREATION
# ============================================================================

print("=" * 70)
print("1. TUPLE CREATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Tuple Literals
# ----------------------------------------------------------------------------
print("\n--- Tuple Literals ---")

# Empty tuple
empty_tuple = ()
print(f"Empty tuple: {empty_tuple}")
print(f"Type: {type(empty_tuple)}")

# Tuple with one element (note the comma!)
single_element = (42,)
print(f"Single element tuple: {single_element}")
print(f"Type: {type(single_element)}")

# Without comma, it's not a tuple
not_a_tuple = (42)
print(f"Without comma: {not_a_tuple}, Type: {type(not_a_tuple)}")  # int

# Tuple with multiple elements
numbers = (1, 2, 3, 4, 5)
print(f"Numbers tuple: {numbers}")

# Mixed types
mixed = (1, "hello", 3.14, True, None)
print(f"Mixed types: {mixed}")

# Nested tuples
nested = ((1, 2), (3, 4), (5, 6))
print(f"Nested tuple: {nested}")

# Tuple with duplicates
duplicates = (1, 2, 2, 3, 3, 3)
print(f"Tuple with duplicates: {duplicates}")


# ----------------------------------------------------------------------------
# 1.2 Tuple Literals Without Parentheses
# ----------------------------------------------------------------------------
print("\n--- Tuple Literals Without Parentheses ---")

# Parentheses are optional (except for empty tuple)
numbers = 1, 2, 3, 4, 5
print(f"Tuple without parentheses: {numbers}")
print(f"Type: {type(numbers)}")

# Single element still needs comma
single = 42,
print(f"Single element: {single}, Type: {type(single)}")

# Multiple assignment is tuple unpacking
a, b, c = 1, 2, 3
print(f"a={a}, b={b}, c={c}")


# ----------------------------------------------------------------------------
# 1.3 Using tuple() Constructor
# ----------------------------------------------------------------------------
print("\n--- Using tuple() Constructor ---")

# From list
list_data = [1, 2, 3]
tuple_from_list = tuple(list_data)
print(f"tuple([1, 2, 3]): {tuple_from_list}")

# From string (iterable)
chars = tuple("hello")
print(f"tuple('hello'): {chars}")

# From range
numbers = tuple(range(5))
print(f"tuple(range(5)): {numbers}")

# From another tuple (creates new tuple)
original = (1, 2, 3)
copied = tuple(original)
print(f"Original: {original}, Copied: {copied}")
print(f"Same object? {original is copied}")  # True for tuples (immutable)


# ----------------------------------------------------------------------------
# 1.4 Tuple vs List Creation
# ----------------------------------------------------------------------------
print("\n--- Tuple vs List Creation ---")

# List uses square brackets
my_list = [1, 2, 3]
print(f"List: {my_list}, Type: {type(my_list)}")

# Tuple uses parentheses (or just commas)
my_tuple = (1, 2, 3)
print(f"Tuple: {my_tuple}, Type: {type(my_tuple)}")

# Both can contain same elements
print(f"Elements same? {list(my_tuple) == my_list}")  # True
print(f"Objects same? {my_tuple == tuple(my_list)}")  # True (by value)


# ============================================================================
# 2. IMMUTABILITY
# ============================================================================

print("\n" + "=" * 70)
print("2. IMMUTABILITY")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Understanding Immutability
# ----------------------------------------------------------------------------
print("\n--- Understanding Immutability ---")

# Tuples are immutable - cannot be modified after creation
my_tuple = (1, 2, 3)
print(f"Original tuple: {my_tuple}")

# Cannot modify elements
# my_tuple[0] = 10  # TypeError: 'tuple' object does not support item assignment

# Cannot add elements
# my_tuple.append(4)  # AttributeError: 'tuple' object has no attribute 'append'

# Cannot remove elements
# my_tuple.remove(2)  # AttributeError: 'tuple' object has no attribute 'remove'

# Lists are mutable - can be modified
my_list = [1, 2, 3]
my_list[0] = 10
my_list.append(4)
print(f"Modified list: {my_list}")


# ----------------------------------------------------------------------------
# 2.2 Creating New Tuples
# ----------------------------------------------------------------------------
print("\n--- Creating New Tuples ---")

# To "modify" a tuple, create a new one
original = (1, 2, 3)
print(f"Original: {original}")

# Add element (create new tuple)
new_tuple = original + (4,)
print(f"After adding 4: {new_tuple}")

# Concatenate tuples
combined = (1, 2) + (3, 4) + (5,)
print(f"Combined: {combined}")

# Repeat tuple
repeated = (1, 2) * 3
print(f"Repeated 3 times: {repeated}")


# ----------------------------------------------------------------------------
# 2.3 Immutability of Nested Structures
# ----------------------------------------------------------------------------
print("\n--- Immutability of Nested Structures ---")

# Tuple containing mutable objects
nested = ([1, 2], [3, 4])
print(f"Nested tuple: {nested}")

# Cannot change the tuple structure
# nested[0] = [5, 6]  # TypeError

# But can modify mutable objects inside
nested[0].append(5)
print(f"After modifying inner list: {nested}")

# The tuple itself is immutable, but contents can be mutable
nested[1][0] = 10
print(f"After modifying inner list element: {nested}")


# ----------------------------------------------------------------------------
# 2.4 When to Use Tuples
# ----------------------------------------------------------------------------
print("\n--- When to Use Tuples ---")

# 1. Fixed collection of items
coordinates = (10, 20)
print(f"Coordinates: {coordinates}")

# 2. Dictionary keys (must be immutable)
locations = {
    (0, 0): "Origin",
    (1, 1): "Corner",
    (2, 2): "Diagonal"
}
print(f"Dictionary with tuple keys: {locations}")

# 3. Return multiple values from function
def get_name_age():
    return "Alice", 25  # Returns tuple

name, age = get_name_age()
print(f"Name: {name}, Age: {age}")

# 4. Data integrity - prevent accidental modification
CONSTANTS = (3.14159, 2.71828, 1.41421)
print(f"Constants: {CONSTANTS}")

# 5. Performance - tuples are slightly faster
import time

# Tuple creation and access is faster
start = time.time()
for _ in range(1000000):
    t = (1, 2, 3)
tuple_time = time.time() - start

start = time.time()
for _ in range(1000000):
    l = [1, 2, 3]
list_time = time.time() - start

print(f"Tuple creation time: {tuple_time:.6f}s")
print(f"List creation time: {list_time:.6f}s")


# ----------------------------------------------------------------------------
# 2.5 When to Use Lists Instead
# ----------------------------------------------------------------------------
print("\n--- When to Use Lists Instead ---")

# Use lists when you need to:
# 1. Modify the collection
items = [1, 2, 3]
items.append(4)
items.remove(2)
print(f"Modified list: {items}")

# 2. Need list-specific methods
numbers = [3, 1, 4, 1, 5]
numbers.sort()
print(f"Sorted list: {numbers}")

# 3. Building collection incrementally
result = []
for i in range(5):
    result.append(i ** 2)
print(f"Built incrementally: {result}")


# ----------------------------------------------------------------------------
# 2.6 Hashability
# ----------------------------------------------------------------------------
print("\n--- Hashability ---")

# Tuples are hashable (if all elements are hashable)
hashable_tuple = (1, 2, 3)
print(f"Hash of {hashable_tuple}: {hash(hashable_tuple)}")

# Can use as dictionary key
my_dict = {hashable_tuple: "value"}
print(f"Dictionary with tuple key: {my_dict}")

# Lists are not hashable
my_list = [1, 2, 3]
# hash(my_list)  # TypeError: unhashable type: 'list'
# {my_list: "value"}  # TypeError: unhashable type: 'list'

# Tuple with unhashable elements is not hashable
unhashable_tuple = ([1, 2], [3, 4])
# hash(unhashable_tuple)  # TypeError: unhashable type: 'list'


# ============================================================================
# 3. UNPACKING
# ============================================================================

print("\n" + "=" * 70)
print("3. UNPACKING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Basic Unpacking
# ----------------------------------------------------------------------------
print("\n--- Basic Unpacking ---")

# Unpack tuple into variables
point = (10, 20)
x, y = point
print(f"Point {point} -> x={x}, y={y}")

# Multiple assignment is tuple unpacking
a, b, c = 1, 2, 3
print(f"a={a}, b={b}, c={c}")

# Swap variables (classic use case)
x, y = 5, 10
print(f"Before swap: x={x}, y={y}")
x, y = y, x  # Swap using tuple unpacking
print(f"After swap: x={x}, y={y}")


# ----------------------------------------------------------------------------
# 3.2 Unpacking from Function Returns
# ----------------------------------------------------------------------------
print("\n--- Unpacking from Function Returns ---")

def get_coordinates():
    """Return x and y coordinates."""
    return 10, 20

# Unpack return value
x, y = get_coordinates()
print(f"Coordinates: x={x}, y={y}")

# Function returning multiple values
def divide_with_remainder(a, b):
    """Return quotient and remainder."""
    return a // b, a % b

quotient, remainder = divide_with_remainder(17, 5)
print(f"17 ÷ 5 = {quotient} remainder {remainder}")


# ----------------------------------------------------------------------------
# 3.3 Extended Unpacking (Python 3+)
# ----------------------------------------------------------------------------
print("\n--- Extended Unpacking (Python 3+) ---")

# Unpack first element and rest
numbers = (1, 2, 3, 4, 5)
first, *rest = numbers
print(f"first={first}, rest={rest}")

# Unpack last element and rest
*beginning, last = numbers
print(f"beginning={beginning}, last={last}")

# Unpack first, middle, and last
first, *middle, last = numbers
print(f"first={first}, middle={middle}, last={last}")

# Multiple rest elements
a, *b, c, d = (1, 2, 3, 4, 5, 6)
print(f"a={a}, b={b}, c={c}, d={d}")


# ----------------------------------------------------------------------------
# 3.4 Unpacking in Loops
# ----------------------------------------------------------------------------
print("\n--- Unpacking in Loops ---")

# List of tuples
points = [(1, 2), (3, 4), (5, 6)]

# Unpack in for loop
print("Unpacking in for loop:")
for x, y in points:
    print(f"  Point: ({x}, {y})")

# With enumerate
print("\nWith enumerate:")
for index, (x, y) in enumerate(points):
    print(f"  Index {index}: ({x}, {y})")

# Dictionary items
person = {"name": "Alice", "age": 25, "city": "NYC"}
print("\nDictionary items:")
for key, value in person.items():
    print(f"  {key}: {value}")


# ----------------------------------------------------------------------------
# 3.5 Unpacking with Ignored Values
# ----------------------------------------------------------------------------
print("\n--- Unpacking with Ignored Values ---")

# Use underscore for ignored values
data = (1, 2, 3, 4, 5)
first, _, third, _, fifth = data
print(f"first={first}, third={third}, fifth={fifth}")

# Ignore multiple values
first, *_, last = data
print(f"first={first}, last={last}")

# Ignore all but one
_, _, _, _, value = data
print(f"Last value: {value}")


# ----------------------------------------------------------------------------
# 3.6 Unpacking in Function Calls
# ----------------------------------------------------------------------------
print("\n--- Unpacking in Function Calls ---")

def greet(name, age, city):
    """Greet a person."""
    print(f"{name}, {age} years old, from {city}")

# Unpack tuple as arguments
person_info = ("Alice", 25, "New York")
greet(*person_info)  # Unpacks tuple as positional arguments

# Unpack dictionary as keyword arguments
person_dict = {"name": "Bob", "age": 30, "city": "London"}
greet(**person_dict)  # Unpacks dict as keyword arguments


# ----------------------------------------------------------------------------
# 3.7 Nested Unpacking
# ----------------------------------------------------------------------------
print("\n--- Nested Unpacking ---")

# Nested tuple
nested = ((1, 2), (3, 4), (5, 6))

# Unpack nested structure
for (x, y), (a, b) in [(nested[0], nested[1])]:
    print(f"First pair: ({x}, {y}), Second pair: ({a}, {b})")

# More complex nested unpacking
data = ((1, 2, 3), (4, 5, 6))
(a, b, c), (d, e, f) = data
print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")


# ============================================================================
# 4. TUPLE METHODS
# ============================================================================

print("\n" + "=" * 70)
print("4. TUPLE METHODS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 count() - Count Occurrences
# ----------------------------------------------------------------------------
print("\n--- count() - Count Occurrences ---")

numbers = (1, 2, 3, 2, 4, 2, 5)
print(f"Tuple: {numbers}")

# Count occurrences of element
count_2 = numbers.count(2)
print(f"count(2) = {count_2}")

count_7 = numbers.count(7)
print(f"count(7) = {count_7}")  # Returns 0 if not found

# Count in mixed tuple
mixed = (1, "hello", 1, True, 1, None)
count_1 = mixed.count(1)
print(f"count(1) in {mixed} = {count_1}")  # Counts both int 1 and True


# ----------------------------------------------------------------------------
# 4.2 index() - Find Index of Element
# ----------------------------------------------------------------------------
print("\n--- index() - Find Index of Element ---")

numbers = (10, 20, 30, 20, 40)
print(f"Tuple: {numbers}")

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
# numbers.index(99)  # ValueError: tuple.index(x): x not in tuple


# ----------------------------------------------------------------------------
# 4.3 Tuple Operations
# ----------------------------------------------------------------------------
print("\n--- Tuple Operations ---")

tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)

# Concatenation
combined = tuple1 + tuple2
print(f"{tuple1} + {tuple2} = {combined}")

# Repetition
repeated = tuple1 * 3
print(f"{tuple1} * 3 = {repeated}")

# Membership
print(f"2 in {tuple1}: {2 in tuple1}")
print(f"5 in {tuple1}: {5 in tuple1}")

# Length
print(f"Length of {tuple1}: {len(tuple1)}")

# Indexing and slicing (same as lists)
print(f"{tuple1}[0] = {tuple1[0]}")
print(f"{tuple1}[-1] = {tuple1[-1]}")
print(f"{tuple1}[1:3] = {tuple1[1:3]}")


# ----------------------------------------------------------------------------
# 4.4 Comparison with Lists
# ----------------------------------------------------------------------------
print("\n--- Comparison with Lists ---")

# Tuples have fewer methods (immutable)
tuple_methods = [method for method in dir(tuple()) if not method.startswith('_')]
list_methods = [method for method in dir([]) if not method.startswith('_')]

print(f"Tuple methods: {tuple_methods}")
print(f"List methods (sample): {list_methods[:10]}...")  # Show first 10

# Common operations
my_tuple = (1, 2, 3)
my_list = [1, 2, 3]

print(f"\nTuple operations:")
print(f"  len({my_tuple}) = {len(my_tuple)}")
print(f"  {my_tuple}[0] = {my_tuple[0]}")
print(f"  2 in {my_tuple} = {2 in my_tuple}")
print(f"  {my_tuple}.count(2) = {my_tuple.count(2)}")
print(f"  {my_tuple}.index(2) = {my_tuple.index(2)}")

print(f"\nList operations (additional):")
print(f"  {my_list}.append(4) -> {my_list}")
print(f"  {my_list}.sort() -> {my_list}")


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Swap two variables
print("\n--- Exercise 1: Swap Variables ---")
a, b = 5, 10
print(f"Before: a={a}, b={b}")
a, b = b, a
print(f"After: a={a}, b={b}")


# Exercise 2: Return multiple values
print("\n--- Exercise 2: Return Multiple Values ---")
def min_max(numbers):
    """Return minimum and maximum."""
    return min(numbers), max(numbers)

numbers = [3, 1, 4, 1, 5, 9, 2, 6]
minimum, maximum = min_max(numbers)
print(f"Numbers: {numbers}")
print(f"Min: {minimum}, Max: {maximum}")


# Exercise 3: Unpack coordinates
print("\n--- Exercise 3: Unpack Coordinates ---")
def distance(p1, p2):
    """Calculate distance between two points."""
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

point1 = (0, 0)
point2 = (3, 4)
dist = distance(point1, point2)
print(f"Distance between {point1} and {point2}: {dist}")


# Exercise 4: Tuple as dictionary key
print("\n--- Exercise 4: Tuple as Dictionary Key ---")
def count_coordinates(points):
    """Count occurrences of each coordinate."""
    coord_count = {}
    for point in points:
        coord_count[point] = coord_count.get(point, 0) + 1
    return coord_count

points = [(1, 2), (3, 4), (1, 2), (5, 6), (3, 4)]
counts = count_coordinates(points)
print(f"Coordinate counts: {counts}")


# Exercise 5: Extended unpacking
print("\n--- Exercise 5: Extended Unpacking ---")
def process_data(data):
    """Process data with extended unpacking."""
    first, *middle, last = data
    return {
        "first": first,
        "middle_sum": sum(middle),
        "last": last
    }

data = (10, 20, 30, 40, 50)
result = process_data(data)
print(f"Data: {data}")
print(f"Result: {result}")


# Exercise 6: Unpacking in function calls
print("\n--- Exercise 6: Unpacking in Function Calls ---")
def create_profile(name, age, city, country):
    """Create a profile."""
    return f"{name}, {age}, from {city}, {country}"

info = ("Alice", 25, "New York", "USA")
profile = create_profile(*info)
print(f"Profile: {profile}")


# Exercise 7: Find all indices of element
print("\n--- Exercise 7: Find All Indices ---")
def find_all_indices(tup, value):
    """Find all indices of value in tuple."""
    indices = []
    start = 0
    while True:
        try:
            idx = tup.index(value, start)
            indices.append(idx)
            start = idx + 1
        except ValueError:
            break
    return indices

numbers = (1, 2, 3, 2, 4, 2, 5)
indices = find_all_indices(numbers, 2)
print(f"Indices of 2 in {numbers}: {indices}")


# Exercise 8: Tuple operations
print("\n--- Exercise 8: Tuple Operations ---")
def tuple_stats(tup):
    """Get statistics about tuple."""
    return {
        "length": len(tup),
        "min": min(tup),
        "max": max(tup),
        "sum": sum(tup) if all(isinstance(x, (int, float)) for x in tup) else None
    }

numbers = (3, 1, 4, 1, 5, 9, 2, 6)
stats = tuple_stats(numbers)
print(f"Stats for {numbers}: {stats}")


# ============================================================================
# 6. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between tuples and lists?
print("\n--- Q1: Tuples vs Lists ---")
print("""
Tuples:
- Immutable (cannot be modified after creation)
- Use parentheses () or just commas
- Fewer methods (count, index)
- Can be used as dictionary keys (if hashable)
- Slightly faster

Lists:
- Mutable (can be modified)
- Use square brackets []
- Many methods (append, extend, sort, etc.)
- Cannot be used as dictionary keys
- More flexible for dynamic data
""")


# Q2: How to create a tuple with one element?
print("\n--- Q2: Single Element Tuple ---")
print("""
Use a comma: (42,) or 42,
Without comma, it's not a tuple: (42) is just int 42
""")


# Q3: Are tuples really immutable?
print("\n--- Q3: Tuple Immutability ---")
print("""
Tuples are immutable - you cannot modify the tuple structure.
However, if a tuple contains mutable objects (like lists),
those objects can still be modified.
The tuple itself cannot be changed, but its contents might be mutable.
""")


# Q4: When should you use tuples vs lists?
print("\n--- Q4: When to Use Tuples vs Lists ---")
print("""
Use tuples when:
- Data should not change (constants, coordinates)
- Need dictionary keys
- Returning multiple values from function
- Performance matters (slightly faster)

Use lists when:
- Need to modify collection
- Need list-specific methods (sort, append, etc.)
- Building collection incrementally
- Order matters and may change
""")


# Q5: What is tuple unpacking?
print("\n--- Q5: Tuple Unpacking ---")
print("""
Tuple unpacking: Assigning tuple elements to multiple variables
Syntax: a, b, c = (1, 2, 3)
Extended unpacking: a, *rest = (1, 2, 3, 4)
Useful for: Swapping variables, function returns, loops
""")


# Q6: Can tuples contain mutable objects?
print("\n--- Q6: Tuples with Mutable Objects ---")
print("""
Yes, tuples can contain mutable objects like lists.
The tuple structure is immutable, but the mutable objects inside can be modified.
Example: t = ([1, 2], [3, 4]) - can modify t[0].append(3)
""")


# Q7: Are tuples hashable?
print("\n--- Q7: Tuple Hashability ---")
print("""
Tuples are hashable IF all elements are hashable.
Hashable types: int, float, str, tuple (if elements hashable)
Not hashable: list, dict, set
Hashable tuples can be used as dictionary keys.
""")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. TUPLE CREATION:
   - Use () or just commas: (1, 2, 3) or 1, 2, 3
   - Single element needs comma: (42,) or 42,
   - Use tuple() constructor to convert from iterables
   - Empty tuple: () (parentheses required)

2. IMMUTABILITY:
   - Tuples cannot be modified after creation
   - Cannot add, remove, or change elements
   - To "modify", create new tuple
   - Immutability applies to structure, not necessarily contents

3. WHEN TO USE TUPLES:
   - Fixed collection of items
   - Dictionary keys (if hashable)
   - Return multiple values from function
   - Data integrity (prevent modification)
   - Performance (slightly faster than lists)

4. WHEN TO USE LISTS:
   - Need to modify collection
   - Need list-specific methods
   - Building collection incrementally
   - Dynamic data that changes

5. UNPACKING:
   - Basic: a, b, c = (1, 2, 3)
   - Extended: first, *rest, last = (1, 2, 3, 4)
   - Use _ for ignored values
   - Useful for swapping, function returns, loops

6. TUPLE METHODS:
   - count(value): Count occurrences
   - index(value): Find first index
   - Limited methods (immutability)
   - Support indexing, slicing, len(), in operator

7. HASHABILITY:
   - Tuples are hashable if all elements are hashable
   - Hashable tuples can be dictionary keys
   - Lists are never hashable

8. COMMON PATTERNS:
   - Swap: a, b = b, a
   - Multiple returns: return x, y
   - Unpack in loops: for x, y in points
   - Dictionary keys: {(x, y): value}

9. BEST PRACTICES:
   - Use tuples for fixed data
   - Use lists for dynamic data
   - Leverage unpacking for clean code
   - Remember immutability when choosing data structure
   - Use tuple unpacking for multiple assignments
""")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
