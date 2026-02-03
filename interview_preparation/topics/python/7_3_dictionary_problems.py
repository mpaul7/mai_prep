"""
Python Dictionary Problems - Interview Preparation
Topic 7.3: Dictionary Problems

This module covers:
- Frequency Analysis: Counting occurrences
- Grouping: Grouping elements by key
- Lookup Problems: Fast lookups, caching
"""

from collections import Counter, defaultdict
from typing import List, Dict, Any
from functools import lru_cache

# ============================================================================
# 1. FREQUENCY ANALYSIS
# ============================================================================

print("=" * 70)
print("1. FREQUENCY ANALYSIS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Basic Frequency Counting
# ----------------------------------------------------------------------------
print("\n--- 1.1 Basic Frequency Counting ---")

def count_frequency_manual(items):
    """Count frequency manually using dictionary."""
    freq = {}
    for item in items:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    return freq

def count_frequency_get(items):
    """Count frequency using dict.get()."""
    freq = {}
    for item in items:
        freq[item] = freq.get(item, 0) + 1
    return freq

def count_frequency_defaultdict(items):
    """Count frequency using defaultdict."""
    freq = defaultdict(int)
    for item in items:
        freq[item] += 1
    return dict(freq)

def count_frequency_counter(items):
    """Count frequency using Counter - most Pythonic."""
    return Counter(items)

items = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
print(f"Items: {items}")
print(f"Frequency (manual): {count_frequency_manual(items)}")
print(f"Frequency (get): {count_frequency_get(items)}")
print(f"Frequency (defaultdict): {count_frequency_defaultdict(items)}")
print(f"Frequency (Counter): {count_frequency_counter(items)}")


# ----------------------------------------------------------------------------
# 1.2 Character Frequency
# ----------------------------------------------------------------------------
print("\n--- 1.2 Character Frequency ---")

def character_frequency(text):
    """Count frequency of each character."""
    return Counter(text)

def character_frequency_ignore_case(text):
    """Count character frequency ignoring case."""
    return Counter(text.lower())

text = "Hello World"
print(f"Character frequency: {character_frequency(text)}")
print(f"Character frequency (ignore case): {character_frequency_ignore_case(text)}")


# ----------------------------------------------------------------------------
# 1.3 Word Frequency
# ----------------------------------------------------------------------------
print("\n--- 1.3 Word Frequency ---")

def word_frequency(text):
    """Count frequency of each word."""
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    return Counter(words)

text = "The quick brown fox jumps over the lazy dog. The dog was quick."
freq = word_frequency(text)
print(f"Word frequency: {freq}")
print(f"Most common: {freq.most_common(3)}")


# ----------------------------------------------------------------------------
# 1.4 Top K Frequent Elements
# ----------------------------------------------------------------------------
print("\n--- 1.4 Top K Frequent Elements ---")

def top_k_frequent_counter(items, k):
    """Find top k frequent elements using Counter."""
    counter = Counter(items)
    return [item for item, count in counter.most_common(k)]

def top_k_frequent_manual(items, k):
    """Find top k frequent elements manually."""
    freq = {}
    for item in items:
        freq[item] = freq.get(item, 0) + 1
    
    # Sort by frequency (descending)
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [item for item, count in sorted_items[:k]]

def top_k_frequent_bucket(items, k):
    """
    Find top k frequent using bucket sort.
    Time: O(n), Space: O(n)
    """
    freq = Counter(items)
    n = len(items)
    
    # Bucket: index = frequency, value = list of items
    buckets = [[] for _ in range(n + 1)]
    for item, count in freq.items():
        buckets[count].append(item)
    
    # Collect top k
    result = []
    for i in range(n, 0, -1):
        for item in buckets[i]:
            result.append(item)
            if len(result) == k:
                return result
    
    return result

items = [1, 1, 1, 2, 2, 3]
k = 2
print(f"Top {k} frequent: {top_k_frequent_counter(items, k)}")
print(f"Top {k} frequent (manual): {top_k_frequent_manual(items, k)}")
print(f"Top {k} frequent (bucket): {top_k_frequent_bucket(items, k)}")


# ----------------------------------------------------------------------------
# 1.5 Frequency Analysis Patterns
# ----------------------------------------------------------------------------
print("\n--- 1.5 Frequency Analysis Patterns ---")

def find_most_frequent(items):
    """Find most frequent element."""
    if not items:
        return None
    counter = Counter(items)
    return counter.most_common(1)[0][0]

def find_least_frequent(items):
    """Find least frequent element."""
    if not items:
        return None
    counter = Counter(items)
    return counter.most_common()[-1][0]

def find_elements_with_frequency(items, target_freq):
    """Find all elements with specific frequency."""
    counter = Counter(items)
    return [item for item, count in counter.items() if count == target_freq]

items = [1, 2, 2, 3, 3, 3, 4]
print(f"Most frequent: {find_most_frequent(items)}")
print(f"Least frequent: {find_least_frequent(items)}")
print(f"Elements with frequency 2: {find_elements_with_frequency(items, 2)}")


# ============================================================================
# 2. GROUPING
# ============================================================================

print("\n" + "=" * 70)
print("2. GROUPING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic Grouping
# ----------------------------------------------------------------------------
print("\n--- 2.1 Basic Grouping ---")

def group_by_key_manual(items, key_func):
    """Group items by key function manually."""
    groups = {}
    for item in items:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups

def group_by_key_defaultdict(items, key_func):
    """Group items using defaultdict."""
    groups = defaultdict(list)
    for item in items:
        key = key_func(item)
        groups[key].append(item)
    return dict(groups)

# Example: Group by first letter
words = ["apple", "banana", "apricot", "berry", "blueberry"]
grouped = group_by_key_defaultdict(words, lambda x: x[0])
print(f"Grouped by first letter: {grouped}")


# ----------------------------------------------------------------------------
# 2.2 Group by Multiple Keys
# ----------------------------------------------------------------------------
print("\n--- 2.2 Group by Multiple Keys ---")

def group_by_multiple_keys(items, key_funcs):
    """Group by multiple keys."""
    groups = defaultdict(list)
    for item in items:
        # Create tuple key from multiple key functions
        key = tuple(f(item) for f in key_funcs)
        groups[key].append(item)
    return dict(groups)

students = [
    {"name": "Alice", "grade": "A", "age": 20},
    {"name": "Bob", "grade": "B", "age": 20},
    {"name": "Charlie", "grade": "A", "age": 21},
    {"name": "David", "grade": "B", "age": 20}
]

grouped = group_by_multiple_keys(
    students,
    [lambda x: x["grade"], lambda x: x["age"]]
)
print(f"Grouped by grade and age: {grouped}")


# ----------------------------------------------------------------------------
# 2.3 Group Anagrams
# ----------------------------------------------------------------------------
print("\n--- 2.3 Group Anagrams ---")

def group_anagrams(words):
    """Group words that are anagrams."""
    groups = defaultdict(list)
    for word in words:
        # Create key from sorted characters
        key = ''.join(sorted(word.lower()))
        groups[key].append(word)
    return list(groups.values())

words = ["eat", "tea", "tan", "ate", "nat", "bat"]
grouped = group_anagrams(words)
print(f"Grouped anagrams: {grouped}")


# ----------------------------------------------------------------------------
# 2.4 Group by Length
# ----------------------------------------------------------------------------
print("\n--- 2.4 Group by Length ---")

def group_by_length(items):
    """Group items by their length."""
    groups = defaultdict(list)
    for item in items:
        groups[len(item)].append(item)
    return dict(groups)

words = ["a", "ab", "abc", "ab", "abcd", "a"]
grouped = group_by_length(words)
print(f"Grouped by length: {grouped}")


# ----------------------------------------------------------------------------
# 2.5 Group by Condition
# ----------------------------------------------------------------------------
print("\n--- 2.5 Group by Condition ---")

def group_by_condition(items, condition_func):
    """Group items by condition (True/False)."""
    groups = defaultdict(list)
    for item in items:
        key = condition_func(item)
        groups[key].append(item)
    return dict(groups)

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grouped = group_by_condition(numbers, lambda x: x % 2 == 0)
print(f"Grouped by even/odd: {grouped}")


# ----------------------------------------------------------------------------
# 2.6 Group and Aggregate
# ----------------------------------------------------------------------------
print("\n--- 2.6 Group and Aggregate ---")

def group_and_sum(items, key_func, value_func):
    """Group items and sum values."""
    groups = defaultdict(int)
    for item in items:
        key = key_func(item)
        value = value_func(item)
        groups[key] += value
    return dict(groups)

sales = [
    {"product": "apple", "amount": 100},
    {"product": "banana", "amount": 150},
    {"product": "apple", "amount": 200},
    {"product": "banana", "amount": 50}
]

total_by_product = group_and_sum(
    sales,
    lambda x: x["product"],
    lambda x: x["amount"]
)
print(f"Total sales by product: {total_by_product}")


# ============================================================================
# 3. LOOKUP PROBLEMS
# ============================================================================

print("\n" + "=" * 70)
print("3. LOOKUP PROBLEMS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Fast Lookups
# ----------------------------------------------------------------------------
print("\n--- 3.1 Fast Lookups ---")

def create_lookup_index(items, key_func):
    """Create lookup index from items."""
    index = {}
    for i, item in enumerate(items):
        key = key_func(item)
        if key not in index:
            index[key] = []
        index[key].append(i)
    return index

students = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
    {"id": 3, "name": "Alice"},
    {"id": 4, "name": "Charlie"}
]

# Create index by name
name_index = create_lookup_index(students, lambda x: x["name"])
print(f"Index by name: {name_index}")
print(f"Find 'Alice': {[students[i] for i in name_index.get('Alice', [])]}")


# ----------------------------------------------------------------------------
# 3.2 Reverse Lookup
# ----------------------------------------------------------------------------
print("\n--- 3.2 Reverse Lookup ---")

def create_reverse_lookup(mapping):
    """Create reverse lookup (value -> key)."""
    reverse = defaultdict(list)
    for key, value in mapping.items():
        reverse[value].append(key)
    return dict(reverse)

mapping = {"a": 1, "b": 2, "c": 1, "d": 3}
reverse = create_reverse_lookup(mapping)
print(f"Original: {mapping}")
print(f"Reverse: {reverse}")


# ----------------------------------------------------------------------------
# 3.3 Caching with Dictionary
# ----------------------------------------------------------------------------
print("\n--- 3.3 Caching with Dictionary ---")

def fibonacci_cached():
    """Fibonacci with manual caching."""
    cache = {}
    
    def fib(n):
        if n in cache:
            return cache[n]
        if n < 2:
            result = n
        else:
            result = fib(n - 1) + fib(n - 2)
        cache[n] = result
        return result
    
    return fib

fib = fibonacci_cached()
print(f"Fibonacci(10): {fib(10)}")
print(f"Fibonacci(20): {fib(20)}")


# ----------------------------------------------------------------------------
# 3.4 Caching with @lru_cache
# ----------------------------------------------------------------------------
print("\n--- 3.4 Caching with @lru_cache ---")

@lru_cache(maxsize=128)
def fibonacci_lru(n):
    """Fibonacci with lru_cache decorator."""
    if n < 2:
        return n
    return fibonacci_lru(n - 1) + fibonacci_lru(n - 2)

print(f"Fibonacci with lru_cache(10): {fibonacci_lru(10)}")
print(f"Cache info: {fibonacci_lru.cache_info()}")


# ----------------------------------------------------------------------------
# 3.5 Memoization Pattern
# ----------------------------------------------------------------------------
print("\n--- 3.5 Memoization Pattern ---")

def memoize(func):
    """Generic memoization decorator."""
    cache = {}
    
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    wrapper.cache = cache
    return wrapper

@memoize
def expensive_function(n):
    """Simulate expensive computation."""
    import time
    time.sleep(0.01)  # Simulate delay
    return n * 2

print(f"First call: {expensive_function(5)}")
print(f"Second call (cached): {expensive_function(5)}")
print(f"Cache: {expensive_function.cache}")


# ----------------------------------------------------------------------------
# 3.6 Two-Level Lookup
# ----------------------------------------------------------------------------
print("\n--- 3.6 Two-Level Lookup ---")

def create_nested_lookup(data, key1_func, key2_func):
    """Create nested lookup structure."""
    lookup = defaultdict(dict)
    for item in data:
        k1 = key1_func(item)
        k2 = key2_func(item)
        lookup[k1][k2] = item
    return dict(lookup)

students = [
    {"name": "Alice", "grade": "A", "subject": "Math"},
    {"name": "Bob", "grade": "B", "subject": "Math"},
    {"name": "Alice", "grade": "A", "subject": "Science"}
]

nested = create_nested_lookup(
    students,
    lambda x: x["name"],
    lambda x: x["subject"]
)
print(f"Nested lookup: {nested}")
print(f"Alice's Math grade: {nested.get('Alice', {}).get('Math', {}).get('grade', 'N/A')}")


# ============================================================================
# 4. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("4. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Most Common Element
print("\n--- Exercise 1: Most Common Element ---")
def most_common_element(items):
    """Find most common element."""
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]

items = [1, 2, 2, 3, 3, 3, 4]
print(f"Most common: {most_common_element(items)}")


# Exercise 2: Group by Category
print("\n--- Exercise 2: Group by Category ---")
def group_by_category(items, category_func):
    """Group items by category."""
    groups = defaultdict(list)
    for item in items:
        groups[category_func(item)].append(item)
    return dict(groups)

products = [
    {"name": "apple", "category": "fruit"},
    {"name": "banana", "category": "fruit"},
    {"name": "carrot", "category": "vegetable"}
]

grouped = group_by_category(products, lambda x: x["category"])
print(f"Grouped by category: {grouped}")


# Exercise 3: Frequency Map Lookup
print("\n--- Exercise 3: Frequency Map Lookup ---")
def create_frequency_map(items):
    """Create frequency map for fast lookups."""
    return Counter(items)

def get_frequency(freq_map, item):
    """Get frequency of item."""
    return freq_map.get(item, 0)

items = ['a', 'b', 'a', 'c', 'b', 'a']
freq_map = create_frequency_map(items)
print(f"Frequency of 'a': {get_frequency(freq_map, 'a')}")
print(f"Frequency of 'x': {get_frequency(freq_map, 'x')}")


# Exercise 4: Group Words by Length
print("\n--- Exercise 4: Group Words by Length ---")
def group_words_by_length(words):
    """Group words by their length."""
    groups = defaultdict(list)
    for word in words:
        groups[len(word)].append(word)
    return dict(groups)

words = ["a", "ab", "abc", "ab", "abcd", "a"]
grouped = group_words_by_length(words)
print(f"Grouped by length: {grouped}")


# Exercise 5: Cached Factorial
print("\n--- Exercise 5: Cached Factorial ---")
@lru_cache(maxsize=100)
def factorial_cached(n):
    """Factorial with caching."""
    if n <= 1:
        return 1
    return n * factorial_cached(n - 1)

print(f"Factorial(10): {factorial_cached(10)}")
print(f"Factorial(5): {factorial_cached(5)}")
print(f"Cache info: {factorial_cached.cache_info()}")


# ============================================================================
# 5. ADVANCED PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("5. ADVANCED PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Frequency-Based Filtering
# ----------------------------------------------------------------------------
print("\n--- 5.1 Frequency-Based Filtering ---")

def filter_by_frequency(items, min_freq=None, max_freq=None):
    """Filter items by frequency range."""
    freq = Counter(items)
    
    if min_freq is not None:
        freq = {k: v for k, v in freq.items() if v >= min_freq}
    
    if max_freq is not None:
        freq = {k: v for k, v in freq.items() if v <= max_freq}
    
    return freq

items = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
filtered = filter_by_frequency(items, min_freq=2, max_freq=3)
print(f"Filtered by frequency [2, 3]: {filtered}")


# ----------------------------------------------------------------------------
# 5.2 Multi-Level Grouping
# ----------------------------------------------------------------------------
print("\n--- 5.2 Multi-Level Grouping ---")

def multi_level_group(items, *key_funcs):
    """Group by multiple levels."""
    if not key_funcs:
        return items
    
    groups = defaultdict(list)
    for item in items:
        key = tuple(f(item) for f in key_funcs)
        groups[key].append(item)
    
    return dict(groups)

students = [
    {"name": "Alice", "grade": "A", "age": 20},
    {"name": "Bob", "grade": "B", "age": 20},
    {"name": "Charlie", "grade": "A", "age": 21}
]

grouped = multi_level_group(
    students,
    lambda x: x["grade"],
    lambda x: x["age"]
)
print(f"Multi-level grouped: {grouped}")


# ----------------------------------------------------------------------------
# 5.3 Inverted Index
# ----------------------------------------------------------------------------
print("\n--- 5.3 Inverted Index ---")

def create_inverted_index(documents):
    """Create inverted index for text search."""
    index = defaultdict(set)
    
    for doc_id, text in enumerate(documents):
        words = text.lower().split()
        for word in words:
            index[word].add(doc_id)
    
    return {word: list(doc_ids) for word, doc_ids in index.items()}

documents = [
    "the quick brown fox",
    "the lazy dog",
    "the quick fox jumps"
]

index = create_inverted_index(documents)
print(f"Inverted index: {index}")
print(f"Documents containing 'quick': {index.get('quick', [])}")


# ============================================================================
# 6. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("6. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. FREQUENCY ANALYSIS:
   - Counter is most Pythonic for counting
   - defaultdict(int) for manual counting
   - most_common(k) for top k elements
   - Bucket sort for O(n) top k

2. GROUPING:
   - defaultdict(list) for grouping
   - Key function determines grouping
   - Can group by multiple keys (tuple)
   - Can group and aggregate simultaneously

3. LOOKUP PROBLEMS:
   - Dictionary provides O(1) average lookup
   - Create index for fast lookups
   - Use caching for expensive computations
   - @lru_cache for automatic memoization

4. COMMON PATTERNS:
   - Counter for frequency analysis
   - defaultdict for grouping
   - Dictionary comprehension for transformations
   - Nested dictionaries for multi-level lookups

5. BEST PRACTICES:
   - Use Counter for counting
   - Use defaultdict for grouping
   - Use @lru_cache for caching
   - Create indices for fast lookups
   - Consider time/space trade-offs

6. PERFORMANCE:
   - Dictionary lookup: O(1) average
   - Counter operations: O(n) for creation
   - Grouping: O(n) time
   - Caching: O(1) lookup after first computation
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Dictionary Problems Guide Ready!")
    print("=" * 70)
