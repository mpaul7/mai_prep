# Python Types: Strings, Collections, and Iteration

This module provides comprehensive coverage of Python's type system, focusing on strings, collections, and iteration patterns commonly tested in data science interviews.

## Overview

Understanding Python's type system is fundamental for data science work. This module covers everything from basic scalar types to advanced collection operations, with emphasis on practical applications and interview-style problems.

## Key Concepts Covered

### 1. Scalar Types
- **int**: Integer numbers with unlimited precision
- **float**: IEEE 754 double precision floating-point numbers
- **str**: Immutable Unicode character sequences
- **bool**: Boolean values (subclass of int)
- **NoneType**: The None singleton

### 2. String Operations
- **Creation and indexing**: String literals, slicing, character access
- **Methods**: upper, lower, split, join, replace, strip, etc.
- **Formatting**: f-strings, format(), % formatting
- **Searching**: find, rfind, count, startswith, endswith
- **Validation**: isalpha, isdigit, isalnum, etc.
- **Regular expressions**: Pattern matching and text processing

### 3. Collections Overview

| Type | Mutable | Ordered | Indexed | Duplicates | Use Case |
|------|---------|---------|---------|------------|----------|
| list | ✅ | ✅ | ✅ | ✅ | General-purpose sequences |
| tuple | ❌ | ✅ | ✅ | ✅ | Immutable sequences, coordinates |
| dict | ✅ | ✅* | ❌ | Keys: ❌, Values: ✅ | Key-value mappings |
| set | ✅ | ❌ | ❌ | ❌ | Unique elements, set operations |

*Ordered since Python 3.7

### 4. Lists (Mutable Sequences)
- **Creation**: Literals, list(), comprehensions
- **Operations**: Indexing, slicing, concatenation
- **Methods**: append, insert, remove, pop, sort, reverse
- **Comprehensions**: `[expr for item in iterable if condition]`
- **Memory**: Dynamic arrays with amortized O(1) append

### 5. Tuples (Immutable Sequences)
- **Creation**: Literals with parentheses, tuple()
- **Unpacking**: Multiple assignment, function arguments
- **Named tuples**: Structured data with named fields
- **Use cases**: Coordinates, database records, function returns

### 6. Dictionaries (Mutable Mappings)
- **Creation**: Literals, dict(), comprehensions
- **Access**: Square brackets, get(), setdefault()
- **Methods**: keys(), values(), items(), update(), pop()
- **Comprehensions**: `{key: value for item in iterable}`
- **Performance**: O(1) average case for access/insertion

### 7. Sets (Mutable Collections)
- **Creation**: Literals with braces, set()
- **Operations**: Union (|), intersection (&), difference (-)
- **Methods**: add(), remove(), discard(), update()
- **Frozen sets**: Immutable version for use as dict keys
- **Applications**: Deduplication, membership testing

### 8. Mutability Concepts
- **Immutable types**: int, float, str, tuple, frozenset
- **Mutable types**: list, dict, set, custom objects
- **Shallow vs deep copy**: copy() vs deepcopy()
- **Common pitfalls**: Mutable default arguments, aliasing

### 9. Iteration Patterns
- **Basic iteration**: for loops, while loops
- **Built-in functions**: enumerate(), zip(), range()
- **Iterators**: iter(), next(), StopIteration
- **Generators**: Generator expressions, yield
- **Advanced**: itertools module, custom iterators

### 10. Advanced Collections
- **Counter**: Frequency counting
- **defaultdict**: Automatic default values
- **deque**: Double-ended queue
- **OrderedDict**: Ordered dictionary (legacy)
- **ChainMap**: Multiple dictionary views

## Files Structure

```
src/types/
├── python_types.py          # Core demonstrations and concepts
├── practice_exercises.py    # Additional practice problems
└── README_types.md         # This documentation
```

## Quick Start

### Basic Usage

```python
from python_types import PythonTypesDemo

# Create demo instance
demo = PythonTypesDemo()

# Run demonstrations
demo.demonstrate_scalar_types()
demo.demonstrate_strings()
demo.demonstrate_lists()
demo.demonstrate_tuples()
demo.demonstrate_dictionaries()
demo.demonstrate_sets()
demo.demonstrate_mutability()
demo.demonstrate_iteration()
demo.demonstrate_advanced_collections()
```

### Practice Exercises

```python
from practice_exercises import PythonTypesExercises

# Create exercises instance
exercises = PythonTypesExercises()

# String exercises
result = exercises.reverse_words_in_string("  hello   world  ")
is_anagram = exercises.is_anagram("listen", "silent")

# Collection exercises
missing = exercises.find_missing_number([3, 0, 1])
merged = exercises.merge_sorted_lists([1, 3, 5], [2, 4, 6])
```

### HackerRank-Style Problems

```python
from python_types import hackerrank_types_problems

# Run practice problems
hackerrank_types_problems()
```

## Common Interview Questions

### 1. "What's the difference between lists and tuples?"

**Answer**:
- **Mutability**: Lists are mutable, tuples are immutable
- **Performance**: Tuples are slightly faster for iteration
- **Use cases**: Lists for changing data, tuples for fixed data
- **Memory**: Tuples use less memory
- **Syntax**: Lists use [], tuples use ()

```python
# List (mutable)
my_list = [1, 2, 3]
my_list.append(4)  # Works

# Tuple (immutable)
my_tuple = (1, 2, 3)
# my_tuple.append(4)  # Error!
```

### 2. "How do dictionaries work internally?"

**Answer**:
- **Hash tables**: Use hash functions for O(1) access
- **Collision handling**: Open addressing with random probing
- **Key requirements**: Must be hashable (immutable)
- **Memory**: Trade space for time efficiency
- **Ordering**: Insertion order preserved since Python 3.7

### 3. "What are the different ways to iterate in Python?"

**Answer**:
```python
data = ['a', 'b', 'c']

# Basic iteration
for item in data:
    print(item)

# With index
for i, item in enumerate(data):
    print(f"{i}: {item}")

# Multiple iterables
for item1, item2 in zip(data1, data2):
    print(item1, item2)

# Dictionary iteration
for key, value in my_dict.items():
    print(f"{key}: {value}")
```

### 4. "Explain shallow vs deep copy"

**Answer**:
```python
import copy

original = [[1, 2], [3, 4]]

# Shallow copy - copies outer structure only
shallow = original.copy()
shallow[0].append(3)  # Affects original!

# Deep copy - copies everything recursively
deep = copy.deepcopy(original)
deep[0].append(4)  # Doesn't affect original
```

### 5. "What are list comprehensions and when to use them?"

**Answer**:
```python
# Traditional approach
result = []
for x in range(10):
    if x % 2 == 0:
        result.append(x**2)

# List comprehension (more Pythonic)
result = [x**2 for x in range(10) if x % 2 == 0]

# Benefits: More readable, often faster, more concise
# When to use: Simple transformations and filtering
# When not to use: Complex logic, side effects
```

## Performance Characteristics

### Time Complexity

| Operation | List | Tuple | Dict | Set |
|-----------|------|-------|------|-----|
| Access by index/key | O(1) | O(1) | O(1) | N/A |
| Search | O(n) | O(n) | O(1) | O(1) |
| Insert/Add | O(1)* | N/A | O(1) | O(1) |
| Delete | O(n) | N/A | O(1) | O(1) |
| Iteration | O(n) | O(n) | O(n) | O(n) |

*Amortized for list append

### Space Complexity

| Type | Memory Usage | Notes |
|------|--------------|-------|
| list | 8 bytes per pointer + object overhead | Dynamic resizing |
| tuple | 8 bytes per pointer + smaller overhead | Fixed size |
| dict | ~3x key-value pairs | Hash table overhead |
| set | ~3x elements | Hash table overhead |

## Best Practices

### Code Style
1. **Use appropriate types**: Choose the right collection for your use case
2. **Prefer comprehensions**: More readable than explicit loops
3. **Handle edge cases**: Empty collections, None values
4. **Use type hints**: Improve code documentation and IDE support

### Performance Tips
1. **List vs tuple**: Use tuples for fixed data
2. **Set membership**: Use sets for fast membership testing
3. **Dictionary lookups**: Prefer dict.get() over try/except
4. **Generator expressions**: Use for large datasets to save memory

### Common Patterns

```python
# Safe dictionary access
value = my_dict.get('key', default_value)

# Counting items
from collections import Counter
counts = Counter(items)

# Grouping items
from collections import defaultdict
groups = defaultdict(list)
for item in items:
    groups[item.category].append(item)

# Flattening nested lists
flat = [item for sublist in nested for item in sublist]

# Dictionary comprehension
squares = {x: x**2 for x in range(10)}

# Set operations for filtering
unique_items = list(set(items))
common_items = set1 & set2
```

## Advanced Topics

### Memory Management
```python
import sys

# Check object size
print(sys.getsizeof([1, 2, 3]))  # List size
print(sys.getsizeof((1, 2, 3)))  # Tuple size

# Memory-efficient iteration
def process_large_file(filename):
    with open(filename) as f:
        for line in f:  # Generator - doesn't load entire file
            yield process_line(line)
```

### Custom Collections
```python
class Stack:
    def __init__(self):
        self._items = []
    
    def push(self, item):
        self._items.append(item)
    
    def pop(self):
        return self._items.pop()
    
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        return iter(reversed(self._items))
```

### Type Checking
```python
from typing import List, Dict, Set, Tuple, Union, Optional

def process_data(
    items: List[str],
    mapping: Dict[str, int],
    options: Optional[Set[str]] = None
) -> Tuple[List[int], Dict[str, str]]:
    # Function with full type annotations
    pass
```

## Practice Problem Categories

### 1. String Manipulation (8 problems)
- Anagram detection
- String compression
- Palindrome checking
- Pattern matching
- Text processing
- Encoding/decoding
- Substring operations
- Regular expressions

### 2. List Operations (10 problems)
- Sorting and searching
- Merging and splitting
- Duplicate handling
- Rotation and reversal
- Two-pointer techniques
- Sliding window
- Dynamic programming
- Array mathematics

### 3. Dictionary Processing (8 problems)
- Frequency counting
- Grouping and aggregation
- Nested dictionary access
- Dictionary merging
- Inversion and transformation
- Caching patterns
- JSON processing
- Configuration management

### 4. Set Operations (6 problems)
- Intersection and union
- Difference operations
- Subset relationships
- Power set generation
- Deduplication
- Mathematical set theory

### 5. Advanced Collections (5 problems)
- Counter applications
- DefaultDict usage
- Deque operations
- Custom data structures
- Memory optimization

## Interview Preparation Strategy

### Study Plan
1. **Week 1**: Master basic types and operations
2. **Week 2**: Practice collection manipulations
3. **Week 3**: Advanced patterns and algorithms
4. **Week 4**: Mock interviews and optimization

### Problem-Solving Approach
1. **Understand requirements**: Ask clarifying questions
2. **Choose appropriate types**: Consider time/space complexity
3. **Start with brute force**: Get working solution first
4. **Optimize**: Improve efficiency if needed
5. **Test edge cases**: Empty inputs, single elements, large data

### Common Mistakes to Avoid
1. **Modifying while iterating**: Use copy or iterate backwards
2. **Mutable default arguments**: Use None and check inside function
3. **Key errors**: Use dict.get() or handle exceptions
4. **Memory leaks**: Be careful with circular references
5. **Type confusion**: Understand mutable vs immutable

## Resources for Further Learning

### Documentation
- [Python Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
- [Built-in Types](https://docs.python.org/3/library/stdtypes.html)
- [Collections Module](https://docs.python.org/3/library/collections.html)

### Books
- "Effective Python" by Brett Slatkin
- "Python Tricks" by Dan Bader
- "Fluent Python" by Luciano Ramalho

### Online Practice
- **LeetCode**: Array, String, Hash Table problems
- **HackerRank**: Python domain challenges
- **Codewars**: Collection manipulation katas

## Contributing

To add new exercises or improve existing ones:

1. Add new methods to appropriate classes
2. Include comprehensive docstrings with examples
3. Provide test cases and expected outputs
4. Update this documentation

## License

This educational material is provided under the MIT License for interview preparation purposes.





