"""
Python Verbal Interview Questions - Interview Preparation
Topic 11.1: Verbal Questions

This module covers:
- Python Concepts: Explain concepts (list vs tuple, mutable vs immutable)
- Code Review: Spot bugs, suggest improvements
- Design Decisions: Why use certain data structures
- Best Practices: Code quality, performance
"""

# ============================================================================
# 1. PYTHON CONCEPTS - EXPLANATIONS
# ============================================================================

print("=" * 70)
print("1. PYTHON CONCEPTS - EXPLANATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 List vs Tuple
# ----------------------------------------------------------------------------
print("\n--- Q: What's the difference between list and tuple? ---")
print("""
LIST:
- Mutable (can be modified after creation)
- Uses square brackets []
- Methods: append(), extend(), remove(), pop(), etc.
- Use when: Need to modify collection, order matters, duplicates allowed
- Example: [1, 2, 3]

TUPLE:
- Immutable (cannot be modified after creation)
- Uses parentheses ()
- Limited methods: count(), index()
- Use when: Fixed collection, dictionary keys, function return values
- Example: (1, 2, 3)

KEY DIFFERENCES:
1. Mutability: Lists can change, tuples cannot
2. Performance: Tuples slightly faster, less memory
3. Use cases: Lists for dynamic data, tuples for fixed data
4. Dictionary keys: Only tuples (immutable) can be keys
""")

# Demonstration
my_list = [1, 2, 3]
my_tuple = (1, 2, 3)

my_list.append(4)  # Works
print(f"List after append: {my_list}")

# my_tuple.append(4)  # Error: tuples are immutable
print(f"Tuple: {my_tuple}")


# ----------------------------------------------------------------------------
# 1.2 Mutable vs Immutable
# ----------------------------------------------------------------------------
print("\n--- Q: Explain mutable vs immutable types ---")
print("""
IMMUTABLE TYPES:
- Cannot be changed after creation
- Operations create new objects
- Examples: int, float, str, bool, tuple, frozenset
- Can be used as dictionary keys
- Thread-safe (no race conditions)

MUTABLE TYPES:
- Can be modified in place
- Operations modify same object
- Examples: list, dict, set
- Cannot be used as dictionary keys
- Need to be careful with references

IMPLICATIONS:
- Immutable: Safer, predictable, but may use more memory
- Mutable: More flexible, efficient for large changes
- Function arguments: Immutable passed by value (conceptually),
  mutable passed by reference (conceptually)
""")

# Demonstration
# Immutable
x = 5
print(f"x id before: {id(x)}")
x = x + 1  # Creates new object
print(f"x id after: {id(x)}")  # Different id

# Mutable
my_list = [1, 2, 3]
print(f"list id before: {id(my_list)}")
my_list.append(4)  # Modifies same object
print(f"list id after: {id(my_list)}")  # Same id


# ----------------------------------------------------------------------------
# 1.3 Dictionary vs List
# ----------------------------------------------------------------------------
print("\n--- Q: When to use dictionary vs list? ---")
print("""
DICTIONARY:
- Key-value pairs
- O(1) average lookup by key
- Unordered (Python 3.7+: insertion order preserved)
- Use when: Need key-value mapping, fast lookups, unique keys
- Example: {"name": "Alice", "age": 25}

LIST:
- Ordered sequence
- O(n) lookup by value
- Indexed by position (0, 1, 2, ...)
- Use when: Need ordered collection, duplicates allowed, sequential access
- Example: [1, 2, 3, 4, 5]

DECISION FACTORS:
- Need key-value pairs? → Dictionary
- Need fast lookup by key? → Dictionary
- Need ordered sequence? → List
- Need duplicates? → List
- Need unique keys? → Dictionary
""")


# ----------------------------------------------------------------------------
# 1.4 Set vs List
# ----------------------------------------------------------------------------
print("\n--- Q: When to use set vs list? ---")
print("""
SET:
- Unordered collection of unique elements
- O(1) average membership test
- No duplicates allowed
- Use when: Need uniqueness, fast membership test, set operations
- Example: {1, 2, 3, 4, 5}

LIST:
- Ordered collection, allows duplicates
- O(n) membership test
- Use when: Need order, duplicates, indexing by position
- Example: [1, 2, 2, 3, 3, 3]

DECISION FACTORS:
- Need uniqueness? → Set
- Need fast membership test? → Set
- Need order? → List
- Need duplicates? → List
- Need set operations (union, intersection)? → Set
""")


# ----------------------------------------------------------------------------
# 1.5 == vs is
# ----------------------------------------------------------------------------
print("\n--- Q: What's the difference between == and is? ---")
print("""
== (Equality):
- Compares VALUES
- Checks if two objects have same content
- Can be overridden with __eq__()
- Example: [1, 2] == [1, 2] → True

is (Identity):
- Compares OBJECT IDENTITY
- Checks if two variables refer to same object
- Cannot be overridden
- Example: [1, 2] is [1, 2] → False (different objects)

USE CASES:
- == for value comparison
- is for None, True, False checks
- is for checking if same object
""")


# ----------------------------------------------------------------------------
# 1.6 Shallow Copy vs Deep Copy
# ----------------------------------------------------------------------------
print("\n--- Q: Explain shallow copy vs deep copy ---")
print("""
SHALLOW COPY:
- Creates new object, but references same nested objects
- Nested objects are shared
- Use copy.copy() or list.copy()
- Faster, less memory

DEEP COPY:
- Creates completely independent copy
- All nested objects are copied recursively
- Use copy.deepcopy()
- Slower, more memory

EXAMPLE:
- Shallow copy: Outer list new, inner lists shared
- Deep copy: Everything is independent
""")


# ----------------------------------------------------------------------------
# 1.7 Generator vs List
# ----------------------------------------------------------------------------
print("\n--- Q: When to use generator vs list? ---")
print("""
GENERATOR:
- Lazy evaluation (generates on demand)
- Memory efficient (O(1) memory)
- Can only iterate once
- Use for: Large datasets, infinite sequences, memory constraints
- Example: (x**2 for x in range(1000000))

LIST:
- Eager evaluation (all values in memory)
- Memory intensive (O(n) memory)
- Can iterate multiple times
- Use for: Small datasets, need multiple iterations, need indexing
- Example: [x**2 for x in range(100)]

DECISION FACTORS:
- Large dataset? → Generator
- Need all values at once? → List
- Memory constrained? → Generator
- Need multiple iterations? → List
""")


# ============================================================================
# 2. CODE REVIEW - SPOT BUGS
# ============================================================================

print("\n" + "=" * 70)
print("2. CODE REVIEW - SPOT BUGS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Bug: Mutable Default Arguments
# ----------------------------------------------------------------------------
print("\n--- Bug 1: Mutable Default Arguments ---")
print("""
BUGGY CODE:
def add_item(item, items=[]):
    items.append(item)
    return items

PROBLEM:
- Default argument [] is created once and shared
- All calls modify the same list
- Unexpected behavior

FIX:
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

EXPLANATION:
- Use None as default, create new list inside function
- Each call gets fresh list
""")

# Demonstration
def buggy_add_item(item, items=[]):
    items.append(item)
    return items

def fixed_add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print("Buggy version:")
result1 = buggy_add_item(1)
result2 = buggy_add_item(2)
print(f"result1: {result1}")  # [1, 2] - Wrong!
print(f"result2: {result2}")  # [1, 2] - Wrong!

print("\nFixed version:")
result1 = fixed_add_item(1)
result2 = fixed_add_item(2)
print(f"result1: {result1}")  # [1] - Correct!
print(f"result2: {result2}")  # [2] - Correct!


# ----------------------------------------------------------------------------
# 2.2 Bug: Modifying List While Iterating
# ----------------------------------------------------------------------------
print("\n--- Bug 2: Modifying List While Iterating ---")
print("""
BUGGY CODE:
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)

PROBLEM:
- Modifying list while iterating causes skipped elements
- Iterator gets confused

FIXES:
1. Iterate over copy:
   for num in numbers[:]:
       if num % 2 == 0:
           numbers.remove(num)

2. Use list comprehension:
   numbers = [num for num in numbers if num % 2 != 0]

3. Iterate backwards:
   for i in range(len(numbers)-1, -1, -1):
       if numbers[i] % 2 == 0:
           numbers.pop(i)
""")


# ----------------------------------------------------------------------------
# 2.3 Bug: String Concatenation in Loop
# ----------------------------------------------------------------------------
print("\n--- Bug 3: String Concatenation in Loop ---")
print("""
BUGGY CODE:
result = ""
for i in range(1000):
    result += str(i)

PROBLEM:
- Strings are immutable, each += creates new string
- O(n²) time complexity
- Inefficient for large strings

FIX:
result = "".join(str(i) for i in range(1000))

EXPLANATION:
- join() is O(n) - much faster
- Creates string once at end
""")


# ----------------------------------------------------------------------------
# 2.4 Bug: Using == for Float Comparison
# ----------------------------------------------------------------------------
print("\n--- Bug 4: Using == for Float Comparison ---")
print("""
BUGGY CODE:
if 0.1 + 0.2 == 0.3:
    print("Equal")

PROBLEM:
- Floating point precision errors
- 0.1 + 0.2 = 0.30000000000000004 (not exactly 0.3)
- == comparison fails

FIX:
import math
if math.isclose(0.1 + 0.2, 0.3):
    print("Equal")

OR:
if abs(0.1 + 0.2 - 0.3) < 1e-9:
    print("Equal")
""")


# ----------------------------------------------------------------------------
# 2.5 Bug: Not Handling Exceptions
# ----------------------------------------------------------------------------
print("\n--- Bug 5: Not Handling Exceptions ---")
print("""
BUGGY CODE:
def divide(a, b):
    return a / b

result = divide(10, 0)  # ZeroDivisionError

PROBLEM:
- No error handling
- Program crashes on invalid input

FIX:
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

OR:
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None  # or raise custom exception
""")


# ----------------------------------------------------------------------------
# 2.6 Bug: Variable Scope Confusion
# ----------------------------------------------------------------------------
print("\n--- Bug 6: Variable Scope Confusion ---")
print("""
BUGGY CODE:
x = 10
def func():
    x = x + 1  # UnboundLocalError
    return x

PROBLEM:
- Assignment to x makes it local
- Can't read local before assignment

FIX:
x = 10
def func():
    global x  # Declare global
    x = x + 1
    return x

OR:
x = 10
def func():
    return x + 1  # Just read, don't assign
""")


# ============================================================================
# 3. DESIGN DECISIONS - DATA STRUCTURES
# ============================================================================

print("\n" + "=" * 70)
print("3. DESIGN DECISIONS - DATA STRUCTURES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 When to Use List
# ----------------------------------------------------------------------------
print("\n--- When to Use List ---")
print("""
USE LIST WHEN:
✓ Need ordered sequence
✓ Need to access by index
✓ Need duplicates
✓ Need to modify frequently (append, remove)
✓ Sequential processing
✓ Small to medium datasets

EXAMPLES:
- Shopping cart items
- Processing queue
- Historical data (time series)
- Results from database query
""")


# ----------------------------------------------------------------------------
# 3.2 When to Use Dictionary
# ----------------------------------------------------------------------------
print("\n--- When to Use Dictionary ---")
print("""
USE DICTIONARY WHEN:
✓ Need key-value mapping
✓ Need fast lookup by key (O(1))
✓ Need unique keys
✓ Need to associate data with identifiers
✓ Configuration settings
✓ Caching/memoization

EXAMPLES:
- User profiles (user_id → user_data)
- Word frequency counting
- Caching function results
- Configuration settings
- API responses (JSON)
""")


# ----------------------------------------------------------------------------
# 3.3 When to Use Set
# ----------------------------------------------------------------------------
print("\n--- When to Use Set ---")
print("""
USE SET WHEN:
✓ Need uniqueness (no duplicates)
✓ Need fast membership test (O(1))
✓ Need set operations (union, intersection, difference)
✓ Don't care about order
✓ Need to remove duplicates

EXAMPLES:
- Tracking visited nodes in graph
- Finding unique elements
- Set operations (union, intersection)
- Removing duplicates from list
- Tag system
""")


# ----------------------------------------------------------------------------
# 3.4 When to Use Tuple
# ----------------------------------------------------------------------------
print("\n--- When to Use Tuple ---")
print("""
USE TUPLE WHEN:
✓ Need immutable sequence
✓ Need dictionary keys
✓ Need to return multiple values from function
✓ Fixed-size collection
✓ Performance critical (slightly faster than list)

EXAMPLES:
- Coordinates (x, y)
- RGB colors (r, g, b)
- Function return values
- Dictionary keys
- Database records (immutable)
""")


# ----------------------------------------------------------------------------
# 3.5 When to Use Generator
# ----------------------------------------------------------------------------
print("\n--- When to Use Generator ---")
print("""
USE GENERATOR WHEN:
✓ Large dataset (doesn't fit in memory)
✓ Infinite sequence
✓ One-time iteration
✓ Memory constrained
✓ Lazy evaluation needed

EXAMPLES:
- Reading large files line by line
- Processing large datasets
- Infinite sequences (Fibonacci)
- Pipeline processing
- Memory-efficient transformations
""")


# ============================================================================
# 4. BEST PRACTICES - CODE QUALITY
# ============================================================================

print("\n" + "=" * 70)
print("4. BEST PRACTICES - CODE QUALITY")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Naming Conventions
# ----------------------------------------------------------------------------
print("\n--- Naming Conventions (PEP 8) ---")
print("""
VARIABLES & FUNCTIONS:
- snake_case: user_name, calculate_total()
- Descriptive names: total_price not tp
- Avoid single letters except for loop variables

CLASSES:
- PascalCase: UserAccount, DataProcessor
- Descriptive names

CONSTANTS:
- UPPER_SNAKE_CASE: MAX_SIZE, DEFAULT_TIMEOUT

PRIVATE:
- Single underscore: _internal_method
- Double underscore: __private_method (name mangling)
""")


# ----------------------------------------------------------------------------
# 4.2 Code Organization
# ----------------------------------------------------------------------------
print("\n--- Code Organization ---")
print("""
PRINCIPLES:
1. Single Responsibility: Each function does one thing
2. DRY (Don't Repeat Yourself): Avoid code duplication
3. Small Functions: Keep functions focused and short
4. Clear Names: Self-documenting code
5. Comments: Explain why, not what
6. Docstrings: Document functions and classes

STRUCTURE:
- Imports at top
- Constants next
- Functions/classes
- Main execution at bottom (if __name__ == "__main__")
""")


# ----------------------------------------------------------------------------
# 4.3 Error Handling
# ----------------------------------------------------------------------------
print("\n--- Error Handling Best Practices ---")
print("""
DO:
✓ Catch specific exceptions
✓ Provide meaningful error messages
✓ Use try-except-finally for cleanup
✓ Log errors appropriately
✓ Handle errors at appropriate level

DON'T:
✗ Use bare except
✗ Suppress exceptions silently
✗ Use exceptions for control flow
✗ Catch too broad exceptions
✗ Ignore error messages
""")


# ----------------------------------------------------------------------------
# 4.4 Performance Best Practices
# ----------------------------------------------------------------------------
print("\n--- Performance Best Practices ---")
print("""
OPTIMIZATIONS:
1. Use appropriate data structures
   - Set for membership: O(1) vs O(n) for list
   - Dict for lookups: O(1) vs O(n) for list

2. Use list comprehensions
   - Faster than loops
   - More Pythonic

3. Avoid premature optimization
   - Profile first, optimize second
   - Readability > micro-optimizations

4. Use generators for large data
   - Memory efficient
   - Lazy evaluation

5. Cache expensive operations
   - Use functools.lru_cache
   - Memoization

COMMON MISTAKES:
- String concatenation in loops
- Unnecessary list copies
- Inefficient algorithms
- Not using built-in functions
""")


# ----------------------------------------------------------------------------
# 4.5 Code Readability
# ----------------------------------------------------------------------------
print("\n--- Code Readability ---")
print("""
IMPROVE READABILITY:
1. Use meaningful variable names
2. Keep functions small (< 20 lines)
3. Use comments for complex logic
4. Use docstrings for functions
5. Follow PEP 8 style guide
6. Use type hints (Python 3.5+)
7. Break complex expressions into steps
8. Use constants for magic numbers

EXAMPLE:
# Bad
if x > 5 and x < 10 and y > 3 and y < 7:
    ...

# Good
MIN_X, MAX_X = 5, 10
MIN_Y, MAX_Y = 3, 7
if MIN_X < x < MAX_X and MIN_Y < y < MAX_Y:
    ...
""")


# ============================================================================
# 5. PERFORMANCE CONSIDERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("5. PERFORMANCE CONSIDERATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Time Complexity Cheat Sheet
# ----------------------------------------------------------------------------
print("\n--- Time Complexity Cheat Sheet ---")
print("""
LIST OPERATIONS:
- Access by index: O(1)
- Append: O(1) amortized
- Insert: O(n)
- Remove: O(n)
- Membership test (in): O(n)
- Sort: O(n log n)

DICTIONARY OPERATIONS:
- Access by key: O(1) average
- Insert: O(1) average
- Delete: O(1) average
- Membership test (in): O(1) average

SET OPERATIONS:
- Add: O(1) average
- Remove: O(1) average
- Membership test (in): O(1) average
- Union/Intersection: O(n + m)

STRING OPERATIONS:
- Concatenation: O(n + m)
- Slicing: O(k) where k is slice length
- Membership test (in): O(n)
""")


# ----------------------------------------------------------------------------
# 5.2 Memory Considerations
# ----------------------------------------------------------------------------
print("\n--- Memory Considerations ---")
print("""
MEMORY EFFICIENT:
- Generators (lazy evaluation)
- Iterators (one item at a time)
- Sets (no duplicates)
- Tuples (slightly less than lists)

MEMORY INTENSIVE:
- Lists (all items in memory)
- String concatenation (creates new strings)
- Deep copies
- Large comprehensions

TIPS:
- Use generators for large datasets
- Process data in chunks
- Delete large objects when done
- Use __slots__ for classes with many instances
""")


# ============================================================================
# 6. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 6.1 Explain Python's GIL
# ----------------------------------------------------------------------------
print("\n--- Q: Explain Python's GIL (Global Interpreter Lock) ---")
print("""
ANSWER:
- GIL is a mutex that protects access to Python objects
- Only one thread executes Python bytecode at a time
- Prevents true parallelism for CPU-bound tasks
- I/O-bound tasks can still benefit from threading
- Multiprocessing bypasses GIL (separate processes)

IMPLICATIONS:
- Threading good for I/O-bound tasks
- Multiprocessing good for CPU-bound tasks
- Async/await for concurrent I/O operations
""")


# ----------------------------------------------------------------------------
# 6.2 How does Python manage memory?
# ----------------------------------------------------------------------------
print("\n--- Q: How does Python manage memory? ---")
print("""
ANSWER:
- Reference counting: Objects deleted when reference count reaches 0
- Garbage collection: Handles circular references
- Memory pools: Pre-allocated memory blocks
- Automatic: No manual memory management needed

KEY POINTS:
- Objects are reference counted
- Circular references handled by GC
- del doesn't immediately free memory
- Memory returned to system when GC runs
""")


# ----------------------------------------------------------------------------
# 6.3 What is duck typing?
# ----------------------------------------------------------------------------
print("\n--- Q: What is duck typing? ---")
print("""
ANSWER:
"If it walks like a duck and quacks like a duck, it's a duck"

- Focus on behavior, not type
- Objects are used based on methods they have
- No explicit type checking needed
- More flexible than strict typing

EXAMPLE:
def process(obj):
    obj.quack()  # Works if obj has quack() method
    # Doesn't care about actual type
""")


# ----------------------------------------------------------------------------
# 6.4 Explain list comprehension vs generator expression
# ----------------------------------------------------------------------------
print("\n--- Q: List comprehension vs generator expression ---")
print("""
LIST COMPREHENSION:
- [x**2 for x in range(10)]
- Creates entire list in memory
- Can iterate multiple times
- Faster for small datasets

GENERATOR EXPRESSION:
- (x**2 for x in range(10))
- Lazy evaluation
- Memory efficient
- Can only iterate once

USE GENERATOR FOR:
- Large datasets
- One-time iteration
- Memory constraints
""")


# ============================================================================
# 7. CODE REVIEW CHECKLIST
# ============================================================================

print("\n" + "=" * 70)
print("7. CODE REVIEW CHECKLIST")
print("=" * 70)

print("""
CODE REVIEW CHECKLIST:

FUNCTIONALITY:
□ Does it work correctly?
□ Are edge cases handled?
□ Are exceptions handled properly?
□ Are return values correct?

CODE QUALITY:
□ Is code readable?
□ Are names descriptive?
□ Are functions small and focused?
□ Is there code duplication?
□ Are comments helpful?

PERFORMANCE:
□ Are appropriate data structures used?
□ Is algorithm efficient?
□ Are there unnecessary operations?
□ Can it be optimized?

STYLE:
□ Follows PEP 8?
□ Consistent formatting?
□ Proper imports?
□ Type hints (if used)?

TESTING:
□ Are edge cases tested?
□ Are error cases tested?
□ Is code testable?
""")


# ============================================================================
# 8. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("8. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. CONCEPT EXPLANATIONS:
   - Be clear and concise
   - Use examples when possible
   - Explain trade-offs
   - Mention use cases

2. CODE REVIEW:
   - Look for common bugs
   - Suggest improvements
   - Explain why changes help
   - Consider edge cases

3. DESIGN DECISIONS:
   - Explain reasoning
   - Consider trade-offs
   - Mention alternatives
   - Justify choices

4. BEST PRACTICES:
   - Follow PEP 8
   - Write readable code
   - Handle errors properly
   - Optimize appropriately

5. COMMUNICATION:
   - Think out loud
   - Ask clarifying questions
   - Explain your reasoning
   - Be open to feedback
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Verbal Questions Guide Ready!")
    print("=" * 70)
    print("\nPractice explaining these concepts out loud.")
    print("Review code examples and identify bugs.")
    print("Think about design decisions and trade-offs.")
