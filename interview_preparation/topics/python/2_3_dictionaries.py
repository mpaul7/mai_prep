"""
Python Dictionaries - Interview Preparation
Topic 2.3: Dictionaries

This module covers:
- Creation: Dict literals, dict(), dict comprehensions
- Access: dict[key], dict.get(), dict.setdefault()
- Methods: keys(), values(), items(), update(), pop(), popitem()
- Dictionary Comprehensions: Basic, conditional
- Nested Dictionaries: Accessing nested values
- DefaultDict: collections.defaultdict
"""

# ============================================================================
# 1. DICTIONARY CREATION
# ============================================================================

print("=" * 70)
print("1. DICTIONARY CREATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Dictionary Literals
# ----------------------------------------------------------------------------
print("\n--- Dictionary Literals ---")

# Empty dictionary
empty_dict = {}
print(f"Empty dict: {empty_dict}")
print(f"Type: {type(empty_dict)}")

# Dictionary with key-value pairs
person = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}
print(f"Person dict: {person}")

# Dictionary with different value types
mixed = {
    "name": "Bob",
    "age": 30,
    "scores": [85, 90, 88],
    "active": True,
    "metadata": None
}
print(f"Mixed types: {mixed}")

# Dictionary with different key types (keys must be hashable)
keys_dict = {
    "string": "value1",
    42: "value2",
    (1, 2): "value3",  # Tuple as key
    True: "value4"
}
print(f"Different key types: {keys_dict}")


# ----------------------------------------------------------------------------
# 1.2 Using dict() Constructor
# ----------------------------------------------------------------------------
print("\n--- Using dict() Constructor ---")

# From keyword arguments
person = dict(name="Alice", age=25, city="New York")
print(f"dict(name='Alice', age=25): {person}")

# From list of tuples
pairs = [("name", "Bob"), ("age", 30), ("city", "London")]
person = dict(pairs)
print(f"dict([('name', 'Bob'), ...]): {person}")

# From two lists using zip
keys = ["name", "age", "city"]
values = ["Charlie", 35, "Tokyo"]
person = dict(zip(keys, values))
print(f"dict(zip(keys, values)): {person}")

# From another dictionary (creates shallow copy)
original = {"a": 1, "b": 2}
copied = dict(original)
print(f"Original: {original}, Copied: {copied}")
print(f"Same object? {original is copied}")  # False


# ----------------------------------------------------------------------------
# 1.3 Dictionary Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Dictionary Comprehensions ---")

# Basic dictionary comprehension
# Syntax: {key: value for item in iterable}
squares = {x: x ** 2 for x in range(5)}
print(f"Squares: {squares}")

# From two lists
keys = ["a", "b", "c"]
values = [1, 2, 3]
mapping = {k: v for k, v in zip(keys, values)}
print(f"Mapping: {mapping}")

# Transform existing dictionary
original = {"a": 1, "b": 2, "c": 3}
doubled = {k: v * 2 for k, v in original.items()}
print(f"Doubled: {doubled}")

# More examples in section 4


# ----------------------------------------------------------------------------
# 1.4 Dictionary Creation Methods Comparison
# ----------------------------------------------------------------------------
print("\n--- Dictionary Creation Methods Comparison ---")

# Method 1: Literal
dict1 = {"a": 1, "b": 2}

# Method 2: dict() constructor
dict2 = dict(a=1, b=2)

# Method 3: dict() from list of tuples
dict3 = dict([("a", 1), ("b", 2)])

# Method 4: Dictionary comprehension
dict4 = {k: v for k, v in [("a", 1), ("b", 2)]}

print(f"All methods create same dict: {dict1 == dict2 == dict3 == dict4}")


# ============================================================================
# 2. ACCESSING DICTIONARY VALUES
# ============================================================================

print("\n" + "=" * 70)
print("2. ACCESSING DICTIONARY VALUES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Using dict[key]
# ----------------------------------------------------------------------------
print("\n--- Using dict[key] ---")

person = {"name": "Alice", "age": 25, "city": "New York"}

# Access value by key
name = person["name"]
print(f"person['name'] = {name}")

age = person["age"]
print(f"person['age'] = {age}")

# KeyError if key doesn't exist
# person["email"]  # KeyError: 'email'

# Access nested dictionary
nested = {"person": {"name": "Bob", "age": 30}}
print(f"nested['person']['name'] = {nested['person']['name']}")


# ----------------------------------------------------------------------------
# 2.2 Using dict.get()
# ----------------------------------------------------------------------------
print("\n--- Using dict.get() ---")

person = {"name": "Alice", "age": 25}

# get() returns None if key doesn't exist (no error)
email = person.get("email")
print(f"person.get('email') = {email}")

# get() with default value
email = person.get("email", "N/A")
print(f"person.get('email', 'N/A') = {email}")

# get() for existing key
name = person.get("name", "Unknown")
print(f"person.get('name', 'Unknown') = {name}")

# get() for nested dictionaries
nested = {"person": {"name": "Bob"}}
name = nested.get("person", {}).get("name", "Unknown")
print(f"Nested get: {name}")


# ----------------------------------------------------------------------------
# 2.3 Using dict.setdefault()
# ----------------------------------------------------------------------------
print("\n--- Using dict.setdefault() ---")

person = {"name": "Alice", "age": 25}

# setdefault() returns value if key exists
name = person.setdefault("name", "Unknown")
print(f"person.setdefault('name', 'Unknown') = {name}")
print(f"Dict after: {person}")

# setdefault() sets and returns default if key doesn't exist
city = person.setdefault("city", "Unknown")
print(f"person.setdefault('city', 'Unknown') = {city}")
print(f"Dict after: {person}")

# Common pattern: Initialize list in dictionary
data = {}
data.setdefault("items", []).append(1)
data.setdefault("items", []).append(2)
print(f"Data with list: {data}")

# Compare with get()
person = {"name": "Alice"}
# get() doesn't modify dictionary
city1 = person.get("city", "Unknown")
print(f"After get(): {person}")

# setdefault() modifies dictionary
city2 = person.setdefault("city", "Unknown")
print(f"After setdefault(): {person}")


# ----------------------------------------------------------------------------
# 2.4 Checking Key Existence
# ----------------------------------------------------------------------------
print("\n--- Checking Key Existence ---")

person = {"name": "Alice", "age": 25}

# Using 'in' operator
print(f"'name' in person: {'name' in person}")
print(f"'email' in person: {'email' in person}")

# Using 'not in'
print(f"'email' not in person: {'email' not in person}")

# Using get() with None check
if person.get("email") is None:
    print("Email not found")

# Using try-except
try:
    email = person["email"]
except KeyError:
    print("Email key not found")


# ----------------------------------------------------------------------------
# 2.5 Access Patterns
# ----------------------------------------------------------------------------
print("\n--- Access Patterns ---")

person = {"name": "Alice", "age": 25}

# Pattern 1: Direct access (raises KeyError if missing)
# name = person["name"]

# Pattern 2: get() with default (safe, doesn't modify)
name = person.get("name", "Unknown")

# Pattern 3: setdefault() (safe, modifies dict if missing)
city = person.setdefault("city", "Unknown")

# Pattern 4: Check before access
if "email" in person:
    email = person["email"]
else:
    email = "N/A"

# Pattern 5: Try-except
try:
    email = person["email"]
except KeyError:
    email = "N/A"


# ============================================================================
# 3. DICTIONARY METHODS
# ============================================================================

print("\n" + "=" * 70)
print("3. DICTIONARY METHODS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 keys() - Get All Keys
# ----------------------------------------------------------------------------
print("\n--- keys() - Get All Keys ---")

person = {"name": "Alice", "age": 25, "city": "New York"}

# Get keys view
keys = person.keys()
print(f"keys(): {keys}")
print(f"Type: {type(keys)}")  # dict_keys view

# Convert to list
keys_list = list(person.keys())
print(f"List of keys: {keys_list}")

# Iterate over keys
print("Iterating keys:")
for key in person.keys():
    print(f"  {key}: {person[key]}")

# Keys view is dynamic
keys_view = person.keys()
person["email"] = "alice@example.com"
print(f"Keys after adding email: {list(keys_view)}")  # Updated automatically


# ----------------------------------------------------------------------------
# 3.2 values() - Get All Values
# ----------------------------------------------------------------------------
print("\n--- values() - Get All Values ---")

person = {"name": "Alice", "age": 25, "city": "New York"}

# Get values view
values = person.values()
print(f"values(): {values}")
print(f"Type: {type(values)}")  # dict_values view

# Convert to list
values_list = list(person.values())
print(f"List of values: {values_list}")

# Iterate over values
print("Iterating values:")
for value in person.values():
    print(f"  {value}")

# Check if value exists
print(f"'Alice' in person.values(): {'Alice' in person.values()}")


# ----------------------------------------------------------------------------
# 3.3 items() - Get Key-Value Pairs
# ----------------------------------------------------------------------------
print("\n--- items() - Get Key-Value Pairs ---")

person = {"name": "Alice", "age": 25, "city": "New York"}

# Get items view
items = person.items()
print(f"items(): {items}")
print(f"Type: {type(items)}")  # dict_items view

# Convert to list
items_list = list(person.items())
print(f"List of items: {items_list}")

# Iterate over items
print("Iterating items:")
for key, value in person.items():
    print(f"  {key}: {value}")

# Unpacking in loop
for key, value in person.items():
    print(f"  {key} = {value}")


# ----------------------------------------------------------------------------
# 3.4 update() - Update Dictionary
# ----------------------------------------------------------------------------
print("\n--- update() - Update Dictionary ---")

person = {"name": "Alice", "age": 25}
print(f"Original: {person}")

# Update with another dictionary
person.update({"city": "New York", "age": 26})
print(f"After update({{'city': 'New York', 'age': 26}}): {person}")

# Update with keyword arguments
person.update(email="alice@example.com", phone="123-456-7890")
print(f"After update(email=..., phone=...): {person}")

# Update with list of tuples
person.update([("country", "USA"), ("zipcode", "10001")])
print(f"After update([('country', 'USA'), ...]): {person}")

# Merge two dictionaries
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
dict1.update(dict2)  # dict2 values overwrite dict1
print(f"Merged: {dict1}")


# ----------------------------------------------------------------------------
# 3.5 pop() - Remove and Return Value
# ----------------------------------------------------------------------------
print("\n--- pop() - Remove and Return Value ---")

person = {"name": "Alice", "age": 25, "city": "New York"}
print(f"Original: {person}")

# pop() with key
age = person.pop("age")
print(f"popped 'age': {age}")
print(f"After pop('age'): {person}")

# pop() with default (no KeyError if key missing)
email = person.pop("email", "N/A")
print(f"popped 'email' (with default): {email}")
print(f"After pop('email', 'N/A'): {person}")

# pop() without default raises KeyError if key missing
# person.pop("email")  # KeyError: 'email'

# Remove last item
person = {"a": 1, "b": 2, "c": 3}
last = person.pop("c")
print(f"Last item: {last}, Remaining: {person}")


# ----------------------------------------------------------------------------
# 3.6 popitem() - Remove and Return Last Item
# ----------------------------------------------------------------------------
print("\n--- popitem() - Remove and Return Last Item ---")

person = {"name": "Alice", "age": 25, "city": "New York"}
print(f"Original: {person}")

# popitem() removes and returns last item (Python 3.7+)
# In Python 3.6 and earlier, removes arbitrary item
key, value = person.popitem()
print(f"popped item: {key}={value}")
print(f"After popitem(): {person}")

# popitem() raises KeyError if dictionary is empty
# empty = {}
# empty.popitem()  # KeyError: 'popitem(): dictionary is empty'


# ----------------------------------------------------------------------------
# 3.7 Other Dictionary Methods
# ----------------------------------------------------------------------------
print("\n--- Other Dictionary Methods ---")

person = {"name": "Alice", "age": 25}

# clear() - Remove all items
person_copy = person.copy()
person_copy.clear()
print(f"After clear(): {person_copy}")

# copy() - Shallow copy
original = {"a": 1, "b": [1, 2, 3]}
copied = original.copy()
print(f"Original: {original}, Copied: {copied}")
print(f"Same object? {original is copied}")  # False
print(f"Nested list same? {original['b'] is copied['b']}")  # True (shallow)

# len() - Get number of items
print(f"Length: {len(person)}")

# del statement - Delete key
person = {"name": "Alice", "age": 25}
del person["age"]
print(f"After del person['age']: {person}")


# ============================================================================
# 4. DICTIONARY COMPREHENSIONS
# ============================================================================

print("\n" + "=" * 70)
print("4. DICTIONARY COMPREHENSIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Basic Dictionary Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Basic Dictionary Comprehensions ---")

# Syntax: {key: value for item in iterable}

# Square numbers
squares = {x: x ** 2 for x in range(5)}
print(f"Squares: {squares}")

# From two lists
keys = ["a", "b", "c"]
values = [1, 2, 3]
mapping = {k: v for k, v in zip(keys, values)}
print(f"Mapping: {mapping}")

# Transform existing dictionary
original = {"a": 1, "b": 2, "c": 3}
doubled = {k: v * 2 for k, v in original.items()}
print(f"Doubled: {doubled}")


# ----------------------------------------------------------------------------
# 4.2 Dictionary Comprehensions with Conditions
# ----------------------------------------------------------------------------
print("\n--- Dictionary Comprehensions with Conditions ---")

# Syntax: {key: value for item in iterable if condition}

# Only even squares
even_squares = {x: x ** 2 for x in range(10) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# Filter dictionary
scores = {"Alice": 85, "Bob": 92, "Charlie": 78, "Diana": 95}
high_scores = {name: score for name, score in scores.items() if score >= 90}
print(f"High scores: {high_scores}")

# Transform keys
data = {"name": "Alice", "age": 25, "city": "NYC"}
upper_keys = {k.upper(): v for k, v in data.items()}
print(f"Upper keys: {upper_keys}")


# ----------------------------------------------------------------------------
# 4.3 Dictionary Comprehensions with if-else
# ----------------------------------------------------------------------------
print("\n--- Dictionary Comprehensions with if-else ---")

# Syntax: {key: (value1 if condition else value2) for item in iterable}

# Mark pass/fail
scores = {"Alice": 85, "Bob": 60, "Charlie": 45, "Diana": 95}
results = {name: "Pass" if score >= 60 else "Fail" for name, score in scores.items()}
print(f"Results: {results}")

# Transform values conditionally
numbers = {"a": 1, "b": 2, "c": 3, "d": 4}
doubled_evens = {k: v * 2 if v % 2 == 0 else v for k, v in numbers.items()}
print(f"Doubled evens: {doubled_evens}")


# ----------------------------------------------------------------------------
# 4.4 Nested Dictionary Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Nested Dictionary Comprehensions ---")

# Create nested dictionary
matrix = {i: {j: i * j for j in range(1, 4)} for i in range(1, 4)}
print(f"Multiplication matrix: {matrix}")

# Flatten nested dictionary
nested = {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}}
flattened = {f"{outer}_{inner}": value 
             for outer, inner_dict in nested.items() 
             for inner, value in inner_dict.items()}
print(f"Flattened: {flattened}")


# ----------------------------------------------------------------------------
# 4.5 Dictionary Comprehensions vs Loops
# ----------------------------------------------------------------------------
print("\n--- Dictionary Comprehensions vs Loops ---")

# Using loop
squares_loop = {}
for x in range(5):
    squares_loop[x] = x ** 2
print(f"Using loop: {squares_loop}")

# Using dictionary comprehension (more Pythonic)
squares_comp = {x: x ** 2 for x in range(5)}
print(f"Using comprehension: {squares_comp}")


# ============================================================================
# 5. NESTED DICTIONARIES
# ============================================================================

print("\n" + "=" * 70)
print("5. NESTED DICTIONARIES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Creating Nested Dictionaries
# ----------------------------------------------------------------------------
print("\n--- Creating Nested Dictionaries ---")

# Nested dictionary
company = {
    "employees": {
        "Alice": {"age": 25, "department": "Engineering"},
        "Bob": {"age": 30, "department": "Sales"}
    },
    "location": {
        "city": "New York",
        "country": "USA"
    }
}
print(f"Nested dictionary: {company}")


# ----------------------------------------------------------------------------
# 5.2 Accessing Nested Values
# ----------------------------------------------------------------------------
print("\n--- Accessing Nested Values ---")

company = {
    "employees": {
        "Alice": {"age": 25, "department": "Engineering"},
        "Bob": {"age": 30, "department": "Sales"}
    }
}

# Direct access
alice_age = company["employees"]["Alice"]["age"]
print(f"Alice's age: {alice_age}")

# Using get() for safe access
bob_dept = company.get("employees", {}).get("Bob", {}).get("department", "Unknown")
print(f"Bob's department: {bob_dept}")

# Safe access with try-except
try:
    charlie_age = company["employees"]["Charlie"]["age"]
except KeyError:
    charlie_age = "Not found"
print(f"Charlie's age: {charlie_age}")


# ----------------------------------------------------------------------------
# 5.3 Modifying Nested Dictionaries
# ----------------------------------------------------------------------------
print("\n--- Modifying Nested Dictionaries ---")

company = {
    "employees": {
        "Alice": {"age": 25, "department": "Engineering"}
    }
}

# Modify existing nested value
company["employees"]["Alice"]["age"] = 26
print(f"After modifying age: {company}")

# Add new nested entry
company["employees"]["Bob"] = {"age": 30, "department": "Sales"}
print(f"After adding Bob: {company}")

# Modify nested structure
company["employees"]["Alice"]["salary"] = 100000
print(f"After adding salary: {company}")


# ----------------------------------------------------------------------------
# 5.4 Iterating Nested Dictionaries
# ----------------------------------------------------------------------------
print("\n--- Iterating Nested Dictionaries ---")

company = {
    "employees": {
        "Alice": {"age": 25, "department": "Engineering"},
        "Bob": {"age": 30, "department": "Sales"}
    }
}

# Iterate outer level
print("Outer level:")
for key, value in company.items():
    print(f"  {key}: {value}")

# Iterate nested level
print("\nNested level:")
for name, info in company["employees"].items():
    print(f"  {name}:")
    for key, value in info.items():
        print(f"    {key}: {value}")


# ----------------------------------------------------------------------------
# 5.5 Flattening Nested Dictionaries
# ----------------------------------------------------------------------------
print("\n--- Flattening Nested Dictionaries ---")

def flatten_dict(d, parent_key="", sep="_"):
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

nested = {
    "person": {
        "name": "Alice",
        "address": {
            "city": "New York",
            "zip": "10001"
        }
    }
}
flattened = flatten_dict(nested)
print(f"Flattened: {flattened}")


# ============================================================================
# 6. DEFAULTDICT
# ============================================================================

print("\n" + "=" * 70)
print("6. DEFAULTDICT")
print("=" * 70)

# ----------------------------------------------------------------------------
# 6.1 Introduction to defaultdict
# ----------------------------------------------------------------------------
print("\n--- Introduction to defaultdict ---")

from collections import defaultdict

# Regular dictionary - KeyError if key doesn't exist
regular_dict = {}
# regular_dict["key"]  # KeyError

# defaultdict - provides default value for missing keys
default_dict = defaultdict(int)  # Default factory: int() returns 0
print(f"defaultdict(int)['missing']: {default_dict['missing']}")
print(f"Dict after: {default_dict}")


# ----------------------------------------------------------------------------
# 6.2 defaultdict with Different Default Factories
# ----------------------------------------------------------------------------
print("\n--- defaultdict with Different Default Factories ---")

# Default to 0 (int)
counts = defaultdict(int)
counts["apple"] += 1
counts["banana"] += 1
counts["apple"] += 1
print(f"Counts: {counts}")

# Default to empty list
groups = defaultdict(list)
groups["fruits"].append("apple")
groups["fruits"].append("banana")
groups["vegetables"].append("carrot")
print(f"Groups: {groups}")

# Default to empty dict
nested = defaultdict(dict)
nested["person"]["name"] = "Alice"
nested["person"]["age"] = 25
print(f"Nested: {nested}")

# Default to empty set
tags = defaultdict(set)
tags["article1"].add("python")
tags["article1"].add("tutorial")
tags["article2"].add("python")
print(f"Tags: {tags}")


# ----------------------------------------------------------------------------
# 6.3 defaultdict with Custom Default Factory
# ----------------------------------------------------------------------------
print("\n--- defaultdict with Custom Default Factory ---")

# Custom function as default factory
def default_value():
    return {"count": 0, "total": 0}

stats = defaultdict(default_value)
stats["Alice"]["count"] += 1
stats["Alice"]["total"] += 100
stats["Bob"]["count"] += 1
print(f"Stats: {stats}")

# Lambda as default factory
default_dict = defaultdict(lambda: "Unknown")
print(f"defaultdict(lambda: 'Unknown')['key']: {default_dict['key']}")


# ----------------------------------------------------------------------------
# 6.4 defaultdict vs Regular Dictionary
# ----------------------------------------------------------------------------
print("\n--- defaultdict vs Regular Dictionary ---")

# Regular dictionary approach
def count_words_regular(words):
    """Count words using regular dict."""
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

# defaultdict approach
def count_words_defaultdict(words):
    """Count words using defaultdict."""
    counts = defaultdict(int)
    for word in words:
        counts[word] += 1  # No need to check if key exists
    return counts

words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
regular_counts = count_words_regular(words)
defaultdict_counts = count_words_defaultdict(words)
print(f"Regular dict: {regular_counts}")
print(f"defaultdict: {defaultdict_counts}")
print(f"Same result: {regular_counts == defaultdict_counts}")


# ----------------------------------------------------------------------------
# 6.5 Converting defaultdict to Regular Dictionary
# ----------------------------------------------------------------------------
print("\n--- Converting defaultdict to Regular Dictionary ---")

dd = defaultdict(int)
dd["a"] = 1
dd["b"] = 2

# Convert to regular dict
regular = dict(dd)
print(f"Regular dict: {regular}, Type: {type(regular)}")

# Or just use it as regular dict (defaultdict is subclass of dict)
print(f"defaultdict is dict: {isinstance(dd, dict)}")


# ----------------------------------------------------------------------------
# 6.6 Common defaultdict Patterns
# ----------------------------------------------------------------------------
print("\n--- Common defaultdict Patterns ---")

# Pattern 1: Grouping
data = [("fruit", "apple"), ("fruit", "banana"), ("vegetable", "carrot")]
groups = defaultdict(list)
for category, item in data:
    groups[category].append(item)
print(f"Grouped: {groups}")

# Pattern 2: Counting
items = ["a", "b", "a", "c", "b", "a"]
counts = defaultdict(int)
for item in items:
    counts[item] += 1
print(f"Counts: {counts}")

# Pattern 3: Nested structures
nested = defaultdict(dict)
nested["user1"]["name"] = "Alice"
nested["user1"]["age"] = 25
nested["user2"]["name"] = "Bob"
print(f"Nested: {nested}")


# ============================================================================
# 7. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Count character frequencies
print("\n--- Exercise 1: Count Character Frequencies ---")
def count_chars(text):
    """Count frequency of each character."""
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    return counts

text = "hello"
print(f"Character counts in '{text}': {count_chars(text)}")


# Exercise 2: Invert dictionary
print("\n--- Exercise 2: Invert Dictionary ---")
def invert_dict(d):
    """Invert key-value pairs."""
    return {v: k for k, v in d.items()}

original = {"a": 1, "b": 2, "c": 3}
inverted = invert_dict(original)
print(f"Original: {original}")
print(f"Inverted: {inverted}")


# Exercise 3: Merge dictionaries
print("\n--- Exercise 3: Merge Dictionaries ---")
def merge_dicts(*dicts):
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        result.update(d)
    return result

dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
dict3 = {"d": 5}
merged = merge_dicts(dict1, dict2, dict3)
print(f"Merged: {merged}")


# Exercise 4: Group by key function
print("\n--- Exercise 4: Group By Key Function ---")
def group_by(items, key_func):
    """Group items by key function."""
    groups = defaultdict(list)
    for item in items:
        key = key_func(item)
        groups[key].append(item)
    return dict(groups)

words = ["apple", "banana", "apricot", "cherry", "date"]
grouped = group_by(words, lambda x: x[0])  # Group by first letter
print(f"Grouped by first letter: {grouped}")


# Exercise 5: Deep access with get
print("\n--- Exercise 5: Deep Access with get ---")
def safe_get(d, *keys, default=None):
    """Safely get nested dictionary value."""
    result = d
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    return result

nested = {"a": {"b": {"c": 1}}}
print(f"safe_get(nested, 'a', 'b', 'c'): {safe_get(nested, 'a', 'b', 'c')}")
print(f"safe_get(nested, 'a', 'x'): {safe_get(nested, 'a', 'x')}")


# Exercise 6: Dictionary comprehension - filter and transform
print("\n--- Exercise 6: Filter and Transform ---")
scores = {"Alice": 85, "Bob": 92, "Charlie": 78, "Diana": 95, "Eve": 88}
high_scores = {name.upper(): score + 5 
               for name, score in scores.items() 
               if score >= 85}
print(f"High scores (transformed): {high_scores}")


# Exercise 7: Word frequency with defaultdict
print("\n--- Exercise 7: Word Frequency ---")
from collections import defaultdict

def word_frequency(text):
    """Count word frequency."""
    words = text.lower().split()
    counts = defaultdict(int)
    for word in words:
        counts[word] += 1
    return dict(counts)

text = "the quick brown fox jumps over the lazy dog"
freq = word_frequency(text)
print(f"Word frequency: {freq}")


# Exercise 8: Nested dictionary operations
print("\n--- Exercise 8: Nested Dictionary Operations ---")
def update_nested(d, keys, value):
    """Update nested dictionary value."""
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value

data = {}
update_nested(data, ["user", "profile", "name"], "Alice")
update_nested(data, ["user", "profile", "age"], 25)
print(f"Nested data: {data}")


# ============================================================================
# 8. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("8. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between dict[key] and dict.get(key)?
print("\n--- Q1: dict[key] vs dict.get(key) ---")
print("""
dict[key]:
- Raises KeyError if key doesn't exist
- Use when you're sure key exists

dict.get(key, default):
- Returns default (None if not specified) if key doesn't exist
- Use for safe access when key might not exist
- Doesn't modify dictionary
""")


# Q2: What's the difference between get() and setdefault()?
print("\n--- Q2: get() vs setdefault() ---")
print("""
get(key, default):
- Returns value or default
- Does NOT modify dictionary

setdefault(key, default):
- Returns value or default
- DOES modify dictionary (sets key=default if missing)
- Useful for initializing nested structures
""")


# Q3: When to use defaultdict?
print("\n--- Q3: When to Use defaultdict ---")
print("""
Use defaultdict when:
- Need default values for missing keys
- Counting/grouping operations
- Avoiding if-else checks for key existence
- Initializing nested structures

Regular dict when:
- Want KeyError for missing keys
- Need explicit control over defaults
- Simpler use cases
""")


# Q4: How do dictionary views work?
print("\n--- Q4: Dictionary Views ---")
print("""
keys(), values(), items() return view objects:
- Dynamic (reflect changes to dictionary)
- Support iteration
- Support membership testing
- Can convert to list: list(dict.keys())
- Memory efficient (don't copy data)
""")


# Q5: How to merge two dictionaries?
print("\n--- Q5: Merging Dictionaries ---")
print("""
Method 1: update()
  dict1.update(dict2)

Method 2: Dictionary unpacking (Python 3.5+)
  {**dict1, **dict2}

Method 3: Dictionary comprehension
  {k: v for d in [dict1, dict2] for k, v in d.items()}
""")


# Q6: Are dictionaries ordered?
print("\n--- Q6: Dictionary Ordering ---")
print("""
Python 3.7+: Dictionaries maintain insertion order
Python 3.6: CPython implementation detail (ordered)
Python < 3.6: Not ordered

Use OrderedDict from collections if you need guaranteed ordering
or compatibility with older Python versions.
""")


# Q7: Can dictionary keys be mutable?
print("\n--- Q7: Mutable Dictionary Keys ---")
print("""
No, dictionary keys must be hashable (immutable).
Hashable types: int, float, str, tuple (if elements hashable)
Not hashable: list, dict, set

Mutable objects cannot be dictionary keys.
""")


# ============================================================================
# 9. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("9. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. DICTIONARY CREATION:
   - Literal: {"key": "value"}
   - dict() constructor: dict(key="value") or dict([(k, v), ...])
   - Dictionary comprehensions: {k: v for k, v in items}
   - From two lists: dict(zip(keys, values))

2. ACCESSING VALUES:
   - dict[key]: Direct access (raises KeyError if missing)
   - dict.get(key, default): Safe access (returns default if missing)
   - dict.setdefault(key, default): Safe access + sets default if missing
   - 'in' operator: Check key existence

3. DICTIONARY METHODS:
   - keys(): Get all keys (view)
   - values(): Get all values (view)
   - items(): Get all key-value pairs (view)
   - update(): Update with another dict
   - pop(key, default): Remove and return value
   - popitem(): Remove and return last item
   - clear(): Remove all items
   - copy(): Shallow copy

4. DICTIONARY COMPREHENSIONS:
   - Basic: {k: v for item in iterable}
   - With condition: {k: v for item in iterable if condition}
   - With if-else: {k: (v1 if cond else v2) for item in iterable}
   - More Pythonic than loops for simple transformations

5. NESTED DICTIONARIES:
   - Access: dict["outer"]["inner"]
   - Safe access: dict.get("outer", {}).get("inner", default)
   - Modify: dict["outer"]["inner"] = value
   - Iterate: Nested loops over items()

6. DEFAULTDICT:
   - Provides default value for missing keys
   - defaultdict(int): Default 0
   - defaultdict(list): Default []
   - defaultdict(dict): Default {}
   - Useful for counting, grouping, nested structures

7. COMMON PATTERNS:
   - Counting: {item: counts.get(item, 0) + 1 for item in items}
   - Grouping: defaultdict(list) with append
   - Merging: {**dict1, **dict2} or dict1.update(dict2)
   - Filtering: {k: v for k, v in d.items() if condition}

8. BEST PRACTICES:
   - Use get() for safe access
   - Use setdefault() when initializing nested structures
   - Use defaultdict for counting/grouping
   - Prefer dictionary comprehensions over loops
   - Use 'in' to check key existence
   - Remember keys must be hashable (immutable)
   - Python 3.7+ maintains insertion order
""")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
