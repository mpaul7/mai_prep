# Python Basics for Data Science Interviews

This module provides comprehensive coverage of Python fundamentals commonly tested in data science interviews and coding assessments.

## Overview

Python is the most popular language for data science, and mastering its fundamentals is crucial for technical interviews. This module covers essential concepts from basic syntax to advanced programming patterns commonly tested in HackerRank-style problems.

## Key Concepts Covered

### 1. Scalar Types
- **int**: Integer numbers (whole numbers)
- **float**: Floating-point numbers (decimals)
- **str**: String (text) data
- **bool**: Boolean values (True/False)
- **Type conversions**: Converting between types

### 2. Operators
- **Arithmetic**: +, -, *, /, //, %, **
- **Comparison**: ==, !=, <, >, <=, >=
- **Logical**: and, or, not
- **Assignment**: =, +=, -=, *=, /=, //=, %=, **=

### 3. Control Flow
- **Conditionals**: if, elif, else statements
- **Loops**: for loops, while loops
- **Loop Control**: break, continue, pass
- **Nested structures**: Nested loops and conditionals

### 4. Functions
- **Definition**: def keyword, parameters, return values
- **Arguments**: Positional, keyword, default parameters
- **Variable arguments**: *args, **kwargs
- **Lambda functions**: Anonymous functions
- **Scope**: Local vs global variables

### 5. Lists and Sequences
- **List operations**: Creation, indexing, slicing
- **List methods**: append, insert, remove, pop, sort
- **List comprehensions**: Concise list creation
- **Multiple list iteration**: zip, enumerate

### 6. String Operations
- **String methods**: upper, lower, split, join, replace
- **String formatting**: f-strings, format(), % formatting
- **String checking**: isdigit, isalpha, isalnum, etc.

## Files Structure

```
src/python/
├── python_basics.py          # Core Python concepts and demonstrations
├── practice_exercises.py     # Additional practice problems
└── README_python.md         # This documentation
```

## Quick Start

### Basic Usage

```python
from python_basics import PythonBasicsDemo

# Create demo instance
demo = PythonBasicsDemo()

# Run demonstrations
demo.demonstrate_scalar_types()
demo.demonstrate_operators()
demo.demonstrate_functions()
demo.demonstrate_conditionals()
demo.demonstrate_loops()
demo.demonstrate_lists()
demo.demonstrate_multiple_list_iteration()
demo.demonstrate_strings()
```

### Practice Exercises

```python
from practice_exercises import PythonPracticeExercises

# Create exercises instance
exercises = PythonPracticeExercises()

# Try different exercises
result = exercises.reverse_string("hello")
is_palindrome = exercises.is_palindrome("racecar")
fibonacci = exercises.fibonacci_sequence(10)
```

### HackerRank-Style Problems

```python
from python_basics import hackerrank_python_problems

# Run practice problems
hackerrank_python_problems()
```

## Common Interview Questions

### 1. "What are the differences between Python data types?"

**Answer**:
- **int**: Whole numbers, unlimited precision in Python 3
- **float**: Decimal numbers, IEEE 754 double precision
- **str**: Immutable sequence of Unicode characters
- **bool**: Subclass of int, True (1) or False (0)
- **Dynamic typing**: Variables can change type during execution

### 2. "Explain list comprehensions vs regular loops"

**Answer**:
```python
# Regular loop
result = []
for x in range(10):
    if x % 2 == 0:
        result.append(x**2)

# List comprehension (more Pythonic)
result = [x**2 for x in range(10) if x % 2 == 0]
```
- **List comprehensions**: More concise, often faster, more readable
- **Regular loops**: More flexible for complex logic

### 3. "How does zip() work with multiple lists?"

**Answer**:
```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
cities = ["NYC", "LA", "Chicago"]

# zip combines elements from multiple iterables
for name, age, city in zip(names, ages, cities):
    print(f"{name}, {age}, {city}")

# zip stops at shortest iterable
# Returns tuples of corresponding elements
```

### 4. "What's the difference between break and continue?"

**Answer**:
- **break**: Exits the entire loop immediately
- **continue**: Skips current iteration, continues with next
- **pass**: Does nothing, placeholder for syntactically required code

### 5. "How do you handle function arguments in Python?"

**Answer**:
```python
def function(pos_arg, default_arg=10, *args, **kwargs):
    # pos_arg: required positional argument
    # default_arg: optional with default value
    # *args: variable positional arguments (tuple)
    # **kwargs: variable keyword arguments (dict)
    pass
```

## Python Fundamentals Reference

### Data Type Operations

| Type | Creation | Common Operations |
|------|----------|-------------------|
| int | `x = 42` | `+, -, *, //, %, **` |
| float | `x = 3.14` | `+, -, *, /, round()` |
| str | `x = "hello"` | `+, *, len(), upper(), split()` |
| bool | `x = True` | `and, or, not` |
| list | `x = [1,2,3]` | `append(), insert(), remove(), sort()` |

### Control Flow Patterns

```python
# Conditional patterns
if condition:
    pass
elif other_condition:
    pass
else:
    pass

# Loop patterns
for item in iterable:
    if condition:
        continue
    if other_condition:
        break
    process(item)

# While loop
while condition:
    # update condition
    pass
```

### Function Patterns

```python
# Basic function
def function_name(param1, param2=default):
    """Docstring describing function"""
    return result

# Lambda function
square = lambda x: x**2

# Multiple return values
def get_stats(data):
    return min(data), max(data), sum(data)/len(data)

min_val, max_val, avg_val = get_stats([1,2,3,4,5])
```

## Practice Problems Categories

### 1. Basic Programming (7 problems)
- FizzBuzz implementation
- List operations and transformations
- String manipulation
- Pattern printing with nested loops
- Grade calculator with conditions
- Student data processing with zip
- Prime number finder with loop control

### 2. String Manipulation (4 problems)
- String reversal without built-ins
- Palindrome detection
- Vowel/consonant counting
- Word frequency analysis

### 3. List Operations (4 problems)
- Duplicate finding
- Merging sorted lists
- List rotation
- Missing number detection

### 4. Mathematical Problems (5 problems)
- Prime number checking
- GCD and LCM calculation
- Factorial computation
- Sum of digits
- Perfect square detection

### 5. Pattern Recognition (2 problems)
- Pascal's triangle generation
- Diamond pattern printing

### 6. Data Processing (3 problems)
- Grouping data by key
- Statistical calculations
- Nested list flattening

## Best Practices for Interviews

### Code Style
1. **Use descriptive variable names**: `student_count` not `sc`
2. **Follow PEP 8**: Consistent spacing and naming
3. **Add docstrings**: Explain function purpose and parameters
4. **Handle edge cases**: Empty inputs, None values, etc.

### Problem-Solving Approach
1. **Understand the problem**: Ask clarifying questions
2. **Plan your solution**: Think before coding
3. **Start simple**: Get basic solution working first
4. **Optimize if needed**: Improve time/space complexity
5. **Test your code**: Walk through examples

### Common Patterns
```python
# Iteration patterns
for i, item in enumerate(items):  # When you need index
for item1, item2 in zip(list1, list2):  # Multiple lists
for key, value in dictionary.items():  # Dictionary iteration

# List comprehensions
result = [expression for item in iterable if condition]

# Dictionary comprehensions
result = {key: value for item in iterable}

# Error handling
try:
    risky_operation()
except SpecificError:
    handle_error()
finally:
    cleanup()
```

## Time Complexity Guide

### Common Operations
- **List access**: O(1)
- **List append**: O(1) amortized
- **List insert**: O(n)
- **List search**: O(n)
- **Dictionary access**: O(1) average
- **Set operations**: O(1) average

### Algorithm Complexities
- **Linear search**: O(n)
- **Binary search**: O(log n)
- **Sorting**: O(n log n)
- **Nested loops**: O(n²)

## Advanced Concepts

### For Senior Interviews
1. **Generators**: Memory-efficient iteration
2. **Decorators**: Function modification
3. **Context managers**: Resource management
4. **List vs Generator comprehensions**: Memory usage
5. **Multiple inheritance**: Method resolution order

### Example Advanced Concepts
```python
# Generator expression
squares = (x**2 for x in range(1000000))  # Memory efficient

# Decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time: {time.time() - start}")
        return result
    return wrapper

# Context manager
with open('file.txt', 'r') as f:
    content = f.read()  # File automatically closed
```

## Tips for Success

### During Interviews
1. **Think out loud**: Explain your reasoning
2. **Start with brute force**: Then optimize
3. **Test with examples**: Walk through your code
4. **Ask about constraints**: Input size, time limits
5. **Consider edge cases**: Empty inputs, single elements

### Common Mistakes to Avoid
1. **Off-by-one errors**: Check loop boundaries
2. **Modifying list while iterating**: Use copy or iterate backwards
3. **Integer division confusion**: Use // for floor division
4. **Mutable default arguments**: Use None and check inside function
5. **Not handling empty inputs**: Always check for edge cases

## Resources for Further Learning

### Online Platforms
- **HackerRank**: Python domain problems
- **LeetCode**: Algorithm and data structure problems
- **Codewars**: Kata challenges
- **Python.org**: Official documentation and tutorial

### Books
- "Automate the Boring Stuff with Python" by Al Sweigart
- "Python Tricks" by Dan Bader
- "Effective Python" by Brett Slatkin

### Practice Strategy
1. **Start with basics**: Master fundamental concepts
2. **Solve daily problems**: Consistent practice
3. **Time yourself**: Simulate interview conditions
4. **Review solutions**: Learn different approaches
5. **Focus on weak areas**: Identify and improve gaps

## Contributing

To add new exercises or improve existing ones:

1. Add new methods to appropriate classes
2. Include comprehensive docstrings
3. Provide examples and test cases
4. Update this documentation

## License

This educational material is provided under the MIT License for interview preparation purposes.
