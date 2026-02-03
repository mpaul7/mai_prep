"""
Python List Comprehensions & Generator Expressions - Interview Preparation
Topic 4.1: List Comprehensions & Generator Expressions

This module covers:
- List Comprehensions: Basic, nested, conditional, multiple loops
- Generator Expressions: Memory-efficient iteration
- Generators: yield keyword, generator functions
"""

import sys
import time

# ============================================================================
# 1. LIST COMPREHENSIONS
# ============================================================================

print("=" * 70)
print("1. LIST COMPREHENSIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Basic List Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Basic List Comprehensions ---")

# Syntax: [expression for item in iterable]

# Square numbers
squares = [x ** 2 for x in range(5)]
print(f"Squares: {squares}")  # [0, 1, 4, 9, 16]

# Double numbers
numbers = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in numbers]
print(f"Doubled: {doubled}")  # [2, 4, 6, 8, 10]

# Convert to strings
strings = [str(x) for x in range(5)]
print(f"Strings: {strings}")  # ['0', '1', '2', '3', '4']

# Extract first character from words
words = ["apple", "banana", "cherry"]
first_chars = [word[0] for word in words]
print(f"First characters: {first_chars}")  # ['a', 'b', 'c']

# Equivalent traditional loop
squares_loop = []
for x in range(5):
    squares_loop.append(x ** 2)
print(f"Using loop: {squares_loop}")


# ----------------------------------------------------------------------------
# 1.2 List Comprehensions with Conditions (Filtering)
# ----------------------------------------------------------------------------
print("\n--- List Comprehensions with Conditions (Filtering) ---")

# Syntax: [expression for item in iterable if condition]

# Even numbers
evens = [x for x in range(10) if x % 2 == 0]
print(f"Even numbers: {evens}")  # [0, 2, 4, 6, 8]

# Numbers greater than 5
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
greater_than_5 = [x for x in numbers if x > 5]
print(f"Numbers > 5: {greater_than_5}")  # [6, 7, 8, 9, 10]

# Words longer than 3 characters
words = ["apple", "pie", "banana", "cat", "cherry"]
long_words = [word for word in words if len(word) > 3]
print(f"Long words: {long_words}")  # ['apple', 'banana', 'cherry']

# Multiple conditions (using and)
divisible_by_2_and_3 = [x for x in range(20) if x % 2 == 0 if x % 3 == 0]
print(f"Divisible by 2 and 3: {divisible_by_2_and_3}")  # [0, 6, 12, 18]

# Alternative: explicit and
divisible_by_2_and_3_alt = [x for x in range(20) if x % 2 == 0 and x % 3 == 0]
print(f"Alternative syntax: {divisible_by_2_and_3_alt}")  # Same result


# ----------------------------------------------------------------------------
# 1.3 List Comprehensions with if-else (Conditional Expression)
# ----------------------------------------------------------------------------
print("\n--- List Comprehensions with if-else (Conditional Expression) ---")

# Syntax: [expression1 if condition else expression2 for item in iterable]

# Mark even/odd
numbers = [1, 2, 3, 4, 5, 6]
parity = ["even" if x % 2 == 0 else "odd" for x in numbers]
print(f"Parity: {parity}")  # ['odd', 'even', 'odd', 'even', 'odd', 'even']

# Square evens, cube odds
transformed = [x ** 2 if x % 2 == 0 else x ** 3 for x in range(1, 6)]
print(f"Transformed: {transformed}")  # [1, 4, 27, 16, 125]

# Replace None with default value
data = [1, None, 3, None, 5]
filled = [x if x is not None else 0 for x in data]
print(f"Filled None values: {filled}")  # [1, 0, 3, 0, 5]

# Multiple conditions with if-elif-else (nested ternary)
scores = [85, 92, 78, 65, 45]
grades = ["A" if s >= 90 else "B" if s >= 80 else "C" if s >= 70 else "D" if s >= 60 else "F" for s in scores]
print(f"Grades: {grades}")  # ['B', 'A', 'C', 'D', 'F']


# ----------------------------------------------------------------------------
# 1.4 Nested List Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Nested List Comprehensions ---")

# Flatten nested list
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
print(f"Flattened: {flattened}")  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Read from inside out: for each sublist in nested, for each item in sublist

# Multiplication table
table = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(f"Multiplication table: {table}")
# [[1, 2, 3], [2, 4, 6], [3, 6, 9]]

# Cartesian product
colors = ["red", "blue"]
sizes = ["S", "M", "L"]
combinations = [(color, size) for color in colors for size in sizes]
print(f"Combinations: {combinations}")
# [('red', 'S'), ('red', 'M'), ('red', 'L'), ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]

# Flatten with condition
nested_numbers = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
evens_flattened = [item for sublist in nested_numbers for item in sublist if item % 2 == 0]
print(f"Even numbers from nested: {evens_flattened}")  # [2, 4, 6, 8]


# ----------------------------------------------------------------------------
# 1.5 Multiple Loops in List Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Multiple Loops in List Comprehensions ---")

# Two loops (like nested for loops)
pairs = [(x, y) for x in range(3) for y in range(2)]
print(f"Pairs: {pairs}")
# [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

# Three loops
triples = [(x, y, z) for x in range(2) for y in range(2) for z in range(2)]
print(f"Triples: {triples}")
# [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

# Using multiple iterables
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
combined = [(x, y) for x in list1 for y in list2]
print(f"Combined: {combined}")
# [(1, 'a'), (1, 'b'), (1, 'c'), (2, 'a'), (2, 'b'), (2, 'c'), (3, 'a'), (3, 'b'), (3, 'c')]

# With condition on both loops
filtered_pairs = [(x, y) for x in range(5) for y in range(5) if x + y == 5]
print(f"Pairs summing to 5: {filtered_pairs}")
# [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]


# ----------------------------------------------------------------------------
# 1.6 List Comprehensions with Functions
# ----------------------------------------------------------------------------
print("\n--- List Comprehensions with Functions ---")

# Using built-in functions
words = ["apple", "pie", "banana"]
lengths = [len(word) for word in words]
print(f"Word lengths: {lengths}")  # [5, 3, 6]

# Using custom functions
def square(x):
    return x ** 2

squares_func = [square(x) for x in range(5)]
print(f"Squares using function: {squares_func}")  # [0, 1, 4, 9, 16]

# Using methods
texts = ["hello", "WORLD", "Python"]
uppercased = [text.upper() for text in texts]
print(f"Uppercased: {uppercased}")  # ['HELLO', 'WORLD', 'PYTHON']


# ----------------------------------------------------------------------------
# 1.7 Dictionary and Set Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Dictionary and Set Comprehensions ---")

# Dictionary comprehension
# Syntax: {key: value for item in iterable}
squares_dict = {x: x ** 2 for x in range(5)}
print(f"Squares dictionary: {squares_dict}")  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Dictionary comprehension with condition
even_squares = {x: x ** 2 for x in range(10) if x % 2 == 0}
print(f"Even squares dict: {even_squares}")  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# Set comprehension
# Syntax: {expression for item in iterable}
unique_squares = {x ** 2 for x in [-2, -1, 0, 1, 2]}
print(f"Unique squares: {unique_squares}")  # {0, 1, 4}

# Set comprehension with condition
unique_evens = {x for x in range(10) if x % 2 == 0}
print(f"Unique evens: {unique_evens}")  # {0, 2, 4, 6, 8}


# ----------------------------------------------------------------------------
# 1.8 When to Use List Comprehensions
# ----------------------------------------------------------------------------
print("\n--- When to Use List Comprehensions ---")

print("""
Advantages:
- More concise and readable (for simple transformations)
- Often faster than equivalent loops
- Pythonic way to create lists

When to use:
- Simple transformations and filtering
- Creating new lists from existing iterables
- When you need the entire list in memory

When NOT to use:
- Complex logic (use regular loops)
- When you need side effects (print, file I/O)
- When readability suffers
- When you don't need the entire list (use generators instead)
""")


# ============================================================================
# 2. GENERATOR EXPRESSIONS
# ============================================================================

print("\n" + "=" * 70)
print("2. GENERATOR EXPRESSIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic Generator Expressions
# ----------------------------------------------------------------------------
print("\n--- Basic Generator Expressions ---")

# Syntax: (expression for item in iterable)
# Note: Uses parentheses instead of square brackets

# Generator expression
squares_gen = (x ** 2 for x in range(5))
print(f"Generator: {squares_gen}")  # <generator object>
print(f"Type: {type(squares_gen)}")  # <class 'generator'>

# Convert to list to see values
print(f"Values: {list(squares_gen)}")  # [0, 1, 4, 9, 16]

# Note: Generator is exhausted after iteration
squares_gen = (x ** 2 for x in range(5))
print(f"First iteration: {list(squares_gen)}")
print(f"Second iteration: {list(squares_gen)}")  # [] (empty, generator exhausted)


# ----------------------------------------------------------------------------
# 2.2 Generator Expressions vs List Comprehensions
# ----------------------------------------------------------------------------
print("\n--- Generator Expressions vs List Comprehensions ---")

# Memory comparison
import sys

# List comprehension - stores all values in memory
list_comp = [x ** 2 for x in range(1000)]
print(f"List comprehension size: {sys.getsizeof(list_comp)} bytes")

# Generator expression - lazy evaluation, minimal memory
gen_exp = (x ** 2 for x in range(1000))
print(f"Generator expression size: {sys.getsizeof(gen_exp)} bytes")

# Generator expressions are much more memory-efficient!


# ----------------------------------------------------------------------------
# 2.3 Generator Expressions with Conditions
# ----------------------------------------------------------------------------
print("\n--- Generator Expressions with Conditions ---")

# Filtering
evens_gen = (x for x in range(10) if x % 2 == 0)
print(f"Even numbers: {list(evens_gen)}")  # [0, 2, 4, 6, 8]

# Conditional expression
parity_gen = ("even" if x % 2 == 0 else "odd" for x in range(5))
print(f"Parity: {list(parity_gen)}")  # ['even', 'odd', 'even', 'odd', 'even']


# ----------------------------------------------------------------------------
# 2.4 Using Generator Expressions
# ----------------------------------------------------------------------------
print("\n--- Using Generator Expressions ---")

# Direct iteration (memory efficient)
print("Iterating generator:")
for square in (x ** 2 for x in range(5)):
    print(square, end=" ")
print()

# With built-in functions that accept iterables
numbers = [1, 2, 3, 4, 5]
sum_of_squares = sum(x ** 2 for x in numbers)
print(f"Sum of squares: {sum_of_squares}")  # 55

max_square = max(x ** 2 for x in numbers)
print(f"Max square: {max_square}")  # 25

# With any() and all()
has_even = any(x % 2 == 0 for x in numbers)
print(f"Has even: {has_even}")  # True

all_positive = all(x > 0 for x in numbers)
print(f"All positive: {all_positive}")  # True


# ----------------------------------------------------------------------------
# 2.5 Generator Expressions in Function Calls
# ----------------------------------------------------------------------------
print("\n--- Generator Expressions in Function Calls ---")

# When parentheses are already present, you can omit extra parentheses
numbers = [1, 2, 3, 4, 5]

# These are equivalent:
sum1 = sum(x ** 2 for x in numbers)
sum2 = sum((x ** 2 for x in numbers))  # Extra parentheses optional
print(f"Sum: {sum1} == {sum2}")  # True

# Useful for large datasets
# Reading large file line by line (memory efficient)
# lines = (line.strip() for line in open('large_file.txt'))


# ----------------------------------------------------------------------------
# 2.6 Nested Generator Expressions
# ----------------------------------------------------------------------------
print("\n--- Nested Generator Expressions ---")

# Flatten nested list (lazy evaluation)
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened_gen = (item for sublist in nested for item in sublist)
print(f"Flattened: {list(flattened_gen)}")  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Cartesian product (lazy)
colors = ["red", "blue"]
sizes = ["S", "M"]
combinations_gen = ((color, size) for color in colors for size in sizes)
print(f"Combinations: {list(combinations_gen)}")
# [('red', 'S'), ('red', 'M'), ('blue', 'S'), ('blue', 'M')]


# ============================================================================
# 3. GENERATOR FUNCTIONS
# ============================================================================

print("\n" + "=" * 70)
print("3. GENERATOR FUNCTIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Basic Generator Functions
# ----------------------------------------------------------------------------
print("\n--- Basic Generator Functions ---")

# Generator function uses 'yield' instead of 'return'
def squares_generator(n):
    """Generate squares from 0 to n-1."""
    for i in range(n):
        yield i ** 2

# Calling generator function returns a generator object
gen = squares_generator(5)
print(f"Generator: {gen}")  # <generator object>
print(f"Values: {list(gen)}")  # [0, 1, 4, 9, 16]

# Generator functions are lazy - values generated on demand
for square in squares_generator(5):
    print(square, end=" ")
print()


# ----------------------------------------------------------------------------
# 3.2 How yield Works
# ----------------------------------------------------------------------------
print("\n--- How yield Works ---")

def count_up_to(max_count):
    """Count from 1 to max_count."""
    count = 1
    while count <= max_count:
        yield count  # Pauses here, returns count, resumes on next()
        count += 1

counter = count_up_to(3)
print(f"First value: {next(counter)}")  # 1
print(f"Second value: {next(counter)}")  # 2
print(f"Third value: {next(counter)}")  # 3
# print(f"Fourth value: {next(counter)}")  # Would raise StopIteration

# Using in loop
for num in count_up_to(3):
    print(num, end=" ")
print()


# ----------------------------------------------------------------------------
# 3.3 Generator Functions vs Regular Functions
# ----------------------------------------------------------------------------
print("\n--- Generator Functions vs Regular Functions ---")

# Regular function - returns all values at once
def squares_list(n):
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

# Generator function - yields values one at a time
def squares_gen_func(n):
    for i in range(n):
        yield i ** 2

# Memory comparison
n = 1000000
# list_result = squares_list(n)  # Creates list with 1M elements
gen_result = squares_gen_func(n)  # Generator object (minimal memory)
print(f"Generator size: {sys.getsizeof(gen_result)} bytes")


# ----------------------------------------------------------------------------
# 3.4 Generator Functions with Multiple yield Statements
# ----------------------------------------------------------------------------
print("\n--- Generator Functions with Multiple yield Statements ---")

def alternating():
    """Yield alternating True/False."""
    while True:
        yield True
        yield False

alt_gen = alternating()
print(f"First 5 values: {[next(alt_gen) for _ in range(5)]}")
# [True, False, True, False, True]


# ----------------------------------------------------------------------------
# 3.5 Generator Functions with Conditions
# ----------------------------------------------------------------------------
print("\n--- Generator Functions with Conditions ---")

def even_numbers(max_num):
    """Generate even numbers up to max_num."""
    for i in range(max_num + 1):
        if i % 2 == 0:
            yield i

evens = list(even_numbers(10))
print(f"Even numbers: {evens}")  # [0, 2, 4, 6, 8, 10]


# ----------------------------------------------------------------------------
# 3.6 Generator Functions with State
# ----------------------------------------------------------------------------
print("\n--- Generator Functions with State ---")

def fibonacci():
    """Generate Fibonacci sequence."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib_gen = fibonacci()
first_10 = [next(fib_gen) for _ in range(10)]
print(f"First 10 Fibonacci numbers: {first_10}")
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


# ----------------------------------------------------------------------------
# 3.7 Sending Values to Generators (Advanced)
# ----------------------------------------------------------------------------
print("\n--- Sending Values to Generators (Advanced) ---")

def accumulator():
    """Accumulator that can receive values via send()."""
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value

acc = accumulator()
next(acc)  # Start generator (prime it)
print(f"Initial total: {acc.send(10)}")  # 10
print(f"After adding 5: {acc.send(5)}")  # 15
print(f"After adding 3: {acc.send(3)}")  # 18


# ----------------------------------------------------------------------------
# 3.8 Generator Pipelines
# ----------------------------------------------------------------------------
print("\n--- Generator Pipelines ---")

def numbers():
    """Generate numbers."""
    for i in range(10):
        yield i

def squares(iterable):
    """Square each number."""
    for x in iterable:
        yield x ** 2

def evens_only(iterable):
    """Filter even numbers."""
    for x in iterable:
        if x % 2 == 0:
            yield x

# Pipeline: numbers -> squares -> evens_only
pipeline = evens_only(squares(numbers()))
result = list(pipeline)
print(f"Pipeline result: {result}")  # [0, 4, 16, 36, 64]


# ============================================================================
# 4. WHEN TO USE GENERATORS
# ============================================================================

print("\n" + "=" * 70)
print("4. WHEN TO USE GENERATORS")
print("=" * 70)

print("""
Use Generators When:
- Working with large datasets (memory efficient)
- Processing data in a pipeline
- Infinite sequences
- When you don't need all values at once
- Streaming data processing

Use List Comprehensions When:
- You need the entire list in memory
- You need to access elements multiple times
- You need list methods (indexing, slicing, etc.)
- Small datasets where memory isn't a concern
""")


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: List comprehension - Extract vowels
print("\n--- Exercise 1: Extract Vowels ---")
def extract_vowels(text):
    return [char for char in text.lower() if char in 'aeiou']

text = "Hello World"
print(f"Vowels in '{text}': {extract_vowels(text)}")  # ['e', 'o', 'o']


# Exercise 2: List comprehension - Transpose matrix
print("\n--- Exercise 2: Transpose Matrix ---")
def transpose_matrix(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

matrix = [[1, 2, 3], [4, 5, 6]]
transposed = transpose_matrix(matrix)
print(f"Original: {matrix}")
print(f"Transposed: {transposed}")  # [[1, 4], [2, 5], [3, 6]]


# Exercise 3: Generator - Prime numbers
print("\n--- Exercise 3: Prime Number Generator ---")
def primes():
    """Generate prime numbers."""
    yield 2
    primes_so_far = [2]
    candidate = 3
    while True:
        is_prime = True
        for prime in primes_so_far:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes_so_far.append(candidate)
            yield candidate
        candidate += 2

prime_gen = primes()
first_10_primes = [next(prime_gen) for _ in range(10)]
print(f"First 10 primes: {first_10_primes}")


# Exercise 4: Generator expression - Sum of squares
print("\n--- Exercise 4: Sum of Squares ---")
def sum_of_squares(n):
    return sum(x ** 2 for x in range(1, n + 1))

print(f"Sum of squares 1-10: {sum_of_squares(10)}")  # 385


# Exercise 5: List comprehension - Flatten and filter
print("\n--- Exercise 5: Flatten and Filter ---")
def flatten_and_filter(nested, condition):
    return [item for sublist in nested for item in sublist if condition(item)]

nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
evens = flatten_and_filter(nested, lambda x: x % 2 == 0)
print(f"Even numbers: {evens}")  # [2, 4, 6, 8]


# Exercise 6: Generator - Reading large file simulation
print("\n--- Exercise 6: Large File Processing ---")
def simulate_large_file(n_lines):
    """Simulate reading large file line by line."""
    for i in range(n_lines):
        yield f"Line {i + 1}"

# Process file without loading all into memory
line_gen = simulate_large_file(1000000)
# Process first 5 lines
for i, line in enumerate(line_gen):
    if i >= 5:
        break
    print(line)


# ============================================================================
# 6. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between list comprehension and generator expression?
print("\n--- Q1: List Comprehension vs Generator Expression ---")
print("""
List Comprehension: [x**2 for x in range(5)]
- Creates entire list in memory
- Can be iterated multiple times
- Faster for small datasets
- Uses square brackets []

Generator Expression: (x**2 for x in range(5))
- Lazy evaluation (generates on demand)
- Memory efficient
- Can only be iterated once
- Uses parentheses ()
""")

# Q2: When would you use a generator function?
print("\n--- Q2: When to Use Generator Functions ---")
print("""
- Large datasets that don't fit in memory
- Infinite sequences
- Processing data in pipelines
- When you need to send values back to generator (send())
- When you need complex logic that's hard to express in comprehension
""")

# Q3: What does yield do?
print("\n--- Q3: What does yield do? ---")
print("""
yield:
- Pauses function execution
- Returns value to caller
- Remembers state (local variables)
- Resumes from where it left off on next call
- Makes function a generator function
""")

# Q4: Can you iterate a generator multiple times?
print("\n--- Q4: Multiple Iterations ---")
gen = (x for x in range(3))
print(f"First iteration: {list(gen)}")  # [0, 1, 2]
print(f"Second iteration: {list(gen)}")  # [] (empty - generator exhausted)

# To iterate multiple times, recreate generator
gen1 = (x for x in range(3))
gen2 = (x for x in range(3))
print(f"Gen1: {list(gen1)}, Gen2: {list(gen2)}")  # Both work


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. LIST COMPREHENSIONS:
   - Syntax: [expression for item in iterable if condition]
   - Can have nested loops: [x for sublist in nested for x in sublist]
   - Can use if-else: [x if condition else y for x in iterable]
   - More Pythonic than loops for simple transformations
   - Creates entire list in memory

2. GENERATOR EXPRESSIONS:
   - Syntax: (expression for item in iterable if condition)
   - Lazy evaluation - generates values on demand
   - Memory efficient for large datasets
   - Can only be iterated once
   - Use when you don't need entire list

3. GENERATOR FUNCTIONS:
   - Use 'yield' instead of 'return'
   - Returns generator object
   - Can maintain state between calls
   - Useful for infinite sequences
   - Can receive values via send()

4. MEMORY EFFICIENCY:
   - List comprehension: O(n) memory
   - Generator expression: O(1) memory
   - Generators are better for large datasets

5. PERFORMANCE:
   - List comprehensions faster for small datasets
   - Generators better for large/infinite datasets
   - Generators enable streaming processing

6. COMMON PATTERNS:
   - Flatten nested: [item for sublist in nested for item in sublist]
   - Filter and transform: [f(x) for x in data if condition(x)]
   - Dictionary comprehension: {k: v for k, v in items}
   - Set comprehension: {x for x in iterable}

7. BEST PRACTICES:
   - Use list comprehensions for simple, readable transformations
   - Use generators for large datasets or pipelines
   - Don't nest too deeply (hurts readability)
   - Prefer generators when memory is a concern
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
