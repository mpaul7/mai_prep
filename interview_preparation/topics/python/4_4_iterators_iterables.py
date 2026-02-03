"""
Python Iterators & Iterables - Interview Preparation
Topic 4.4: Iterators & Iterables

This module covers:
- Iterables: Objects that can be iterated
- Iterators: __iter__, __next__
- Built-in Functions: iter(), next(), enumerate(), zip()
"""

# ============================================================================
# 1. UNDERSTANDING ITERABLES AND ITERATORS
# ============================================================================

print("=" * 70)
print("1. UNDERSTANDING ITERABLES AND ITERATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 What is an Iterable?
# ----------------------------------------------------------------------------
print("\n--- What is an Iterable? ---")

print("""
An ITERABLE is an object that can return an iterator.
It implements __iter__() method which returns an iterator.

Common iterables:
- Lists, tuples, strings, dictionaries, sets
- range(), files, generators
- Any object with __iter__() method
""")

# Examples of iterables
iterables = [
    [1, 2, 3],           # List
    (1, 2, 3),           # Tuple
    "hello",             # String
    {"a": 1, "b": 2},    # Dictionary
    {1, 2, 3},           # Set
    range(5),            # Range
]

for item in iterables:
    print(f"{type(item).__name__}: {item}")


# ----------------------------------------------------------------------------
# 1.2 What is an Iterator?
# ----------------------------------------------------------------------------
print("\n--- What is an Iterator? ---")

print("""
An ITERATOR is an object that implements:
- __iter__() method (returns self)
- __next__() method (returns next item, raises StopIteration when done)

Iterators are stateful - they remember their position.
Once exhausted, they cannot be reused.
""")

# Getting iterator from iterable
my_list = [1, 2, 3]
my_iterator = iter(my_list)
print(f"List: {my_list}")
print(f"Iterator: {my_iterator}")
print(f"Iterator type: {type(my_iterator)}")


# ----------------------------------------------------------------------------
# 1.3 Iterable vs Iterator
# ----------------------------------------------------------------------------
print("\n--- Iterable vs Iterator ---")

# List is iterable
my_list = [1, 2, 3]
print(f"List is iterable: {hasattr(my_list, '__iter__')}")

# Get iterator from list
my_iterator = iter(my_list)
print(f"Iterator has __iter__: {hasattr(my_iterator, '__iter__')}")
print(f"Iterator has __next__: {hasattr(my_iterator, '__next__')}")

# Iterator is also iterable (returns self)
print(f"Iterator is iterable: {hasattr(my_iterator, '__iter__')}")

# Key difference: Iterator can be exhausted
print("\nIterating through iterator:")
for item in my_iterator:
    print(item, end=" ")
print()

# Iterator is now exhausted
print(f"Trying to iterate again (exhausted): {list(my_iterator)}")  # []

# But iterable can create new iterator
new_iterator = iter(my_list)
print(f"New iterator: {list(new_iterator)}")  # [1, 2, 3]


# ============================================================================
# 2. BUILT-IN FUNCTIONS
# ============================================================================

print("\n" + "=" * 70)
print("2. BUILT-IN FUNCTIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 iter() - Get Iterator from Iterable
# ----------------------------------------------------------------------------
print("\n--- iter() - Get Iterator from Iterable ---")

# Basic usage
my_list = [1, 2, 3, 4, 5]
iterator = iter(my_list)
print(f"Iterator from list: {iterator}")

# Get next value
print(f"First value: {next(iterator)}")  # 1
print(f"Second value: {next(iterator)}")  # 2

# iter() with sentinel value (advanced)
# iter(callable, sentinel) - calls callable until it returns sentinel
import random

def random_number():
    return random.randint(1, 10)

# Stop when random_number() returns 5
iterator = iter(random_number, 5)
print("Random numbers until 5:")
for num in iterator:
    print(num, end=" ")
    if num == 5:
        break
print()


# ----------------------------------------------------------------------------
# 2.2 next() - Get Next Item from Iterator
# ----------------------------------------------------------------------------
print("\n--- next() - Get Next Item from Iterator ---")

# Basic usage
my_list = [10, 20, 30]
iterator = iter(my_list)

print(f"next(iterator): {next(iterator)}")  # 10
print(f"next(iterator): {next(iterator)}")  # 20
print(f"next(iterator): {next(iterator)}")  # 30

# next() with default value (when iterator exhausted)
iterator = iter([1, 2])
print(f"next(iterator): {next(iterator)}")  # 1
print(f"next(iterator): {next(iterator)}")  # 2
print(f"next(iterator, 'No more items'): {next(iterator, 'No more items')}")  # Default

# Without default, raises StopIteration
iterator = iter([1])
next(iterator)
try:
    next(iterator)  # Raises StopIteration
except StopIteration:
    print("StopIteration raised when iterator exhausted")


# ----------------------------------------------------------------------------
# 2.3 enumerate() - Add Index to Iterable
# ----------------------------------------------------------------------------
print("\n--- enumerate() - Add Index to Iterable ---")

# Basic usage
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"Index {index}: {fruit}")

# enumerate() returns iterator of tuples
enum_iter = enumerate(fruits)
print(f"enumerate() returns: {list(enum_iter)}")
# [(0, 'apple'), (1, 'banana'), (2, 'cherry')]

# Start index from different number
for index, fruit in enumerate(fruits, start=1):
    print(f"Position {index}: {fruit}")

# Common use case: Finding index while iterating
target = "banana"
for index, fruit in enumerate(fruits):
    if fruit == target:
        print(f"Found '{target}' at index {index}")
        break


# ----------------------------------------------------------------------------
# 2.4 zip() - Combine Multiple Iterables
# ----------------------------------------------------------------------------
print("\n--- zip() - Combine Multiple Iterables ---")

# Basic usage - combines corresponding elements
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

zipped = zip(names, ages)
print(f"zip() returns: {list(zipped)}")
# [('Alice', 25), ('Bob', 30), ('Charlie', 35)]

# Iterating through zipped result
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# zip() stops at shortest iterable
list1 = [1, 2, 3]
list2 = ['a', 'b']
zipped = zip(list1, list2)
print(f"zip() with different lengths: {list(zipped)}")
# [(1, 'a'), (2, 'b')] - stops at shortest

# Unzipping (using zip with *)
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
numbers, letters = zip(*pairs)
print(f"Unzipped numbers: {numbers}")  # (1, 2, 3)
print(f"Unzipped letters: {letters}")  # ('a', 'b', 'c')

# Multiple iterables
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
list3 = [True, False, True]
for a, b, c in zip(list1, list2, list3):
    print(f"({a}, {b}, {c})")


# ============================================================================
# 3. MAKING OBJECTS ITERABLE
# ============================================================================

print("\n" + "=" * 70)
print("3. MAKING OBJECTS ITERABLE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Simple Iterable Class
# ----------------------------------------------------------------------------
print("\n--- Simple Iterable Class ---")

class NumberRange:
    """Simple iterable class."""
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        """Return iterator."""
        return NumberRangeIterator(self.start, self.end)

class NumberRangeIterator:
    """Iterator for NumberRange."""
    
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# Using the iterable
number_range = NumberRange(1, 5)
print(f"NumberRange(1, 5): {list(number_range)}")  # [1, 2, 3, 4]

# Can iterate multiple times (creates new iterator each time)
for num in number_range:
    print(num, end=" ")
print()

for num in number_range:  # Works again!
    print(num, end=" ")
print()


# ----------------------------------------------------------------------------
# 3.2 Iterable and Iterator Combined
# ----------------------------------------------------------------------------
print("\n--- Iterable and Iterator Combined ---")

class Counter:
    """Class that is both iterable and iterator."""
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.current = start
    
    def __iter__(self):
        """Return iterator (self in this case)."""
        return self
    
    def __next__(self):
        """Return next value."""
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

counter = Counter(1, 4)
print(f"Counter(1, 4): {list(counter)}")  # [1, 2, 3]

# Note: Can only iterate once (iterator exhausted)
print(f"Trying again: {list(counter)}")  # [] - exhausted!


# ----------------------------------------------------------------------------
# 3.3 Iterable with __getitem__ (Alternative)
# ----------------------------------------------------------------------------
print("\n--- Iterable with __getitem__ (Alternative) ---")

class SimpleIterable:
    """Iterable using __getitem__ (old-style, but still works)."""
    
    def __init__(self, *items):
        self.items = list(items)
    
    def __getitem__(self, index):
        """Get item by index."""
        if index >= len(self.items):
            raise IndexError
        return self.items[index]
    
    def __len__(self):
        return len(self.items)

# Python automatically creates iterator from __getitem__
simple = SimpleIterable(1, 2, 3, 4, 5)
print(f"SimpleIterable: {list(simple)}")  # [1, 2, 3, 4, 5]

# Works with for loop
for item in simple:
    print(item, end=" ")
print()


# ============================================================================
# 4. CUSTOM ITERATORS
# ============================================================================

print("\n" + "=" * 70)
print("4. CUSTOM ITERATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Fibonacci Iterator
# ----------------------------------------------------------------------------
print("\n--- Fibonacci Iterator ---")

class FibonacciIterator:
    """Iterator that generates Fibonacci numbers."""
    
    def __init__(self, max_count):
        self.max_count = max_count
        self.count = 0
        self.a = 0
        self.b = 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count >= self.max_count:
            raise StopIteration
        
        if self.count == 0:
            result = self.a
        elif self.count == 1:
            result = self.b
        else:
            result = self.a + self.b
            self.a, self.b = self.b, result
        
        self.count += 1
        return result

fib = FibonacciIterator(10)
print(f"First 10 Fibonacci numbers: {list(fib)}")
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


# ----------------------------------------------------------------------------
# 4.2 Reverse Iterator
# ----------------------------------------------------------------------------
print("\n--- Reverse Iterator ---")

class ReverseIterator:
    """Iterator that iterates in reverse."""
    
    def __init__(self, iterable):
        self.items = list(iterable)
        self.index = len(self.items) - 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < 0:
            raise StopIteration
        value = self.items[self.index]
        self.index -= 1
        return value

reverse = ReverseIterator([1, 2, 3, 4, 5])
print(f"Reversed: {list(reverse)}")  # [5, 4, 3, 2, 1]


# ----------------------------------------------------------------------------
# 4.3 Even Numbers Iterator
# ----------------------------------------------------------------------------
print("\n--- Even Numbers Iterator ---")

class EvenNumbers:
    """Iterable that generates even numbers."""
    
    def __init__(self, max_value):
        self.max_value = max_value
    
    def __iter__(self):
        return EvenNumbersIterator(self.max_value)

class EvenNumbersIterator:
    """Iterator for even numbers."""
    
    def __init__(self, max_value):
        self.current = 0
        self.max_value = max_value
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current > self.max_value:
            raise StopIteration
        value = self.current
        self.current += 2
        return value

evens = EvenNumbers(10)
print(f"Even numbers up to 10: {list(evens)}")  # [0, 2, 4, 6, 8, 10]


# ----------------------------------------------------------------------------
# 4.4 Infinite Iterator
# ----------------------------------------------------------------------------
print("\n--- Infinite Iterator ---")

class InfiniteCounter:
    """Iterator that counts infinitely."""
    
    def __init__(self, start=0, step=1):
        self.current = start
        self.step = step
    
    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.current
        self.current += self.step
        return value

# Be careful with infinite iterators!
counter = InfiniteCounter(0, 1)
print("First 5 values from infinite counter:")
for i, value in enumerate(counter):
    if i >= 5:
        break
    print(value, end=" ")
print()


# ============================================================================
# 5. ADVANCED PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("5. ADVANCED PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Iterator Protocol with Generators
# ----------------------------------------------------------------------------
print("\n--- Iterator Protocol with Generators ---")

# Generators automatically implement iterator protocol
def fibonacci_generator(max_count):
    """Generator that yields Fibonacci numbers."""
    a, b = 0, 1
    count = 0
    while count < max_count:
        yield a
        a, b = b, a + b
        count += 1

fib_gen = fibonacci_generator(5)
print(f"Fibonacci generator: {list(fib_gen)}")  # [0, 1, 1, 2, 3]

# Generator is an iterator
print(f"Has __iter__: {hasattr(fib_gen, '__iter__')}")
print(f"Has __next__: {hasattr(fib_gen, '__next__')}")


# ----------------------------------------------------------------------------
# 5.2 Chaining Iterators
# ----------------------------------------------------------------------------
print("\n--- Chaining Iterators ---")

from itertools import chain

list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

# Chain multiple iterables
chained = chain(list1, list2, list3)
print(f"Chained: {list(chained)}")  # [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ----------------------------------------------------------------------------
# 5.3 Checking if Object is Iterable
# ----------------------------------------------------------------------------
print("\n--- Checking if Object is Iterable ---")

def is_iterable(obj):
    """Check if object is iterable."""
    try:
        iter(obj)
        return True
    except TypeError:
        return False

print(f"List is iterable: {is_iterable([1, 2, 3])}")  # True
print(f"String is iterable: {is_iterable('hello')}")  # True
print(f"Integer is iterable: {is_iterable(42)}")  # False
print(f"None is iterable: {is_iterable(None)}")  # False


# ----------------------------------------------------------------------------
# 5.4 Manual Iteration
# ----------------------------------------------------------------------------
print("\n--- Manual Iteration ---")

my_list = [1, 2, 3, 4, 5]
iterator = iter(my_list)

# Manual iteration
try:
    while True:
        value = next(iterator)
        print(value, end=" ")
except StopIteration:
    print("\nIteration complete")


# ============================================================================
# 6. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Square Numbers Iterator
print("\n--- Exercise 1: Square Numbers Iterator ---")

class SquareNumbers:
    """Iterable that generates square numbers."""
    
    def __init__(self, max_value):
        self.max_value = max_value
    
    def __iter__(self):
        return SquareNumbersIterator(self.max_value)

class SquareNumbersIterator:
    """Iterator for square numbers."""
    
    def __init__(self, max_value):
        self.current = 0
        self.max_value = max_value
    
    def __iter__(self):
        return self
    
    def __next__(self):
        square = self.current ** 2
        if square > self.max_value:
            raise StopIteration
        result = square
        self.current += 1
        return result

squares = SquareNumbers(25)
print(f"Squares up to 25: {list(squares)}")  # [0, 1, 4, 9, 16, 25]


# Exercise 2: Range Iterator
print("\n--- Exercise 2: Range Iterator ---")

class MyRange:
    """Custom range iterator."""
    
    def __init__(self, start, stop=None, step=1):
        if stop is None:
            self.start = 0
            self.stop = start
        else:
            self.start = start
            self.stop = stop
        self.step = step
    
    def __iter__(self):
        return MyRangeIterator(self.start, self.stop, self.step)

class MyRangeIterator:
    """Iterator for MyRange."""
    
    def __init__(self, start, stop, step):
        self.current = start
        self.stop = stop
        self.step = step
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.step > 0 and self.current >= self.stop) or \
           (self.step < 0 and self.current <= self.stop):
            raise StopIteration
        value = self.current
        self.current += self.step
        return value

my_range = MyRange(1, 10, 2)
print(f"MyRange(1, 10, 2): {list(my_range)}")  # [1, 3, 5, 7, 9]


# Exercise 3: Pairwise Iterator
print("\n--- Exercise 3: Pairwise Iterator ---")

def pairwise(iterable):
    """Return pairs of consecutive elements."""
    iterator = iter(iterable)
    try:
        prev = next(iterator)
    except StopIteration:
        return
    
    for item in iterator:
        yield (prev, item)
        prev = item

numbers = [1, 2, 3, 4, 5]
pairs = list(pairwise(numbers))
print(f"Pairwise of {numbers}: {pairs}")  # [(1, 2), (2, 3), (3, 4), (4, 5)]


# Exercise 4: Batch Iterator
print("\n--- Exercise 4: Batch Iterator ---")

class BatchIterator:
    """Iterator that yields items in batches."""
    
    def __init__(self, iterable, batch_size):
        self.iterator = iter(iterable)
        self.batch_size = batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = []
        try:
            for _ in range(self.batch_size):
                batch.append(next(self.iterator))
        except StopIteration:
            if not batch:
                raise
        return batch

numbers = list(range(10))
batches = list(BatchIterator(numbers, 3))
print(f"Batches of 3: {batches}")  # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


# ============================================================================
# 7. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("7. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between iterable and iterator?
print("\n--- Q1: Iterable vs Iterator ---")
print("""
Iterable:
- Object that can return an iterator
- Implements __iter__() method
- Can create multiple iterators
- Examples: list, tuple, string, dict

Iterator:
- Object that implements __iter__() and __next__()
- Stateful - remembers position
- Can only be iterated once (exhausted)
- Iterator is also iterable (returns self)
""")

# Q2: How does for loop work?
print("\n--- Q2: How does for loop work? ---")
print("""
for item in iterable:
    # code

Is equivalent to:
iterator = iter(iterable)
while True:
    try:
        item = next(iterator)
        # code
    except StopIteration:
        break
""")

# Q3: What happens when iterator is exhausted?
print("\n--- Q3: What happens when iterator is exhausted? ---")
print("""
When iterator is exhausted:
- __next__() raises StopIteration exception
- for loop catches StopIteration and exits
- Iterator cannot be reused (must create new one)
- next() with default value returns default instead of raising
""")

# Q4: Can you iterate over a dictionary?
print("\n--- Q4: Iterating over Dictionary ---")
my_dict = {"a": 1, "b": 2, "c": 3}

print("Iterating keys (default):")
for key in my_dict:
    print(key)

print("Iterating values:")
for value in my_dict.values():
    print(value)

print("Iterating items:")
for key, value in my_dict.items():
    print(f"{key}: {value}")


# Q5: What's the difference between zip() and enumerate()?
print("\n--- Q5: zip() vs enumerate() ---")
print("""
zip():
- Combines multiple iterables element-wise
- Returns tuples of corresponding elements
- Stops at shortest iterable
- Example: zip([1,2], ['a','b']) -> [(1,'a'), (2,'b')]

enumerate():
- Adds index to single iterable
- Returns (index, value) tuples
- Starts from 0 (or specified start)
- Example: enumerate(['a','b']) -> [(0,'a'), (1,'b')]
""")


# ============================================================================
# 8. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("8. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. ITERABLE:
   - Object that can return an iterator
   - Implements __iter__() method
   - Can create multiple iterators
   - Examples: list, tuple, string, dict, set, range

2. ITERATOR:
   - Object that implements __iter__() and __next__()
   - Stateful - remembers position
   - Can only be iterated once (exhausted)
   - Iterator is also iterable (returns self)

3. ITERATOR PROTOCOL:
   - __iter__(): Returns iterator (or self)
   - __next__(): Returns next item, raises StopIteration when done
   - StopIteration: Exception raised when iterator exhausted

4. BUILT-IN FUNCTIONS:
   - iter(obj): Get iterator from iterable
   - next(iterator, default): Get next item, default if exhausted
   - enumerate(iterable, start=0): Add index to iterable
   - zip(*iterables): Combine multiple iterables element-wise

5. MAKING OBJECTS ITERABLE:
   - Implement __iter__() that returns iterator
   - Or implement __getitem__() (old-style, but works)
   - Iterator should implement __iter__() and __next__()

6. COMMON PATTERNS:
   - Separate iterable and iterator classes (allows multiple iterations)
   - Combined iterable/iterator (simpler, but can only iterate once)
   - Generator functions (automatically implement iterator protocol)

7. BEST PRACTICES:
   - Use generators for simple iterators
   - Separate iterable and iterator for reusable iteration
   - Handle StopIteration properly
   - Use enumerate() when you need index
   - Use zip() to combine multiple sequences
   - Be careful with infinite iterators

8. COMMON MISTAKES:
   - Trying to iterate exhausted iterator again
   - Not handling StopIteration
   - Modifying iterable while iterating
   - Confusing iterable with iterator
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
