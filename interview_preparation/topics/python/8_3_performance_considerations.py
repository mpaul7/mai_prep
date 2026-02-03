"""
Python Performance Considerations - Interview Preparation
Topic 8.3: Performance Considerations

This module covers:
- Time Complexity: Big O notation basics
- Space Complexity: Memory usage
- List vs Set: When to use which
- Comprehensions vs Loops: Performance trade-offs
"""

import time
import sys
from typing import List, Set

# ============================================================================
# 1. TIME COMPLEXITY - BIG O NOTATION BASICS
# ============================================================================

print("=" * 70)
print("1. TIME COMPLEXITY - BIG O NOTATION BASICS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Understanding Big O Notation
# ----------------------------------------------------------------------------
print("\n--- 1.1 Understanding Big O Notation ---")
print("""
BIG O NOTATION describes how runtime grows with input size:
- O(1): Constant time - same time regardless of input size
- O(log n): Logarithmic - time grows slowly (binary search)
- O(n): Linear - time grows proportionally with input
- O(n log n): Linearithmic - common for efficient sorting
- O(n²): Quadratic - nested loops, time grows quadratically
- O(2ⁿ): Exponential - very slow, avoid if possible
- O(n!): Factorial - extremely slow

COMMON COMPLEXITIES:
- O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)
""")


# ----------------------------------------------------------------------------
# 1.2 Common Operations Time Complexity
# ----------------------------------------------------------------------------
print("\n--- 1.2 Common Operations Time Complexity ---")

# O(1) - Constant Time Operations
print("\nO(1) - Constant Time:")
print("  - List access by index: my_list[0]")
print("  - Dictionary access by key: my_dict['key']")
print("  - Set membership test: element in my_set")
print("  - Append to list: my_list.append(x)")

# O(n) - Linear Time Operations
print("\nO(n) - Linear Time:")
print("  - Iterating through list: for item in my_list")
print("  - List membership test: element in my_list")
print("  - Finding max/min: max(my_list)")
print("  - String concatenation: 'a' + 'b' + 'c'")

# O(n log n) - Linearithmic Time Operations
print("\nO(n log n) - Linearithmic Time:")
print("  - Sorting: sorted(my_list)")
print("  - Mergesort, Heapsort")

# O(n²) - Quadratic Time Operations
print("\nO(n²) - Quadratic Time:")
print("  - Nested loops: for i in range(n): for j in range(n)")
print("  - List insert at beginning: my_list.insert(0, x)")
print("  - List remove: my_list.remove(x)")
print("  - Bubble sort, Selection sort")


# ----------------------------------------------------------------------------
# 1.3 Demonstrating Time Complexity
# ----------------------------------------------------------------------------
print("\n--- 1.3 Demonstrating Time Complexity ---")

def demonstrate_o1():
    """O(1) - Constant time"""
    my_list = list(range(1000))
    my_dict = {i: i for i in range(1000)}
    my_set = set(range(1000))
    
    # All O(1) operations
    _ = my_list[500]      # List access
    _ = my_dict[500]      # Dict access
    _ = 500 in my_set     # Set membership
    
    print("  O(1) operations complete - constant time regardless of size")

def demonstrate_on(n):
    """O(n) - Linear time"""
    my_list = list(range(n))
    count = 0
    
    # O(n) - iterate through list
    for item in my_list:
        count += 1
    
    # O(n) - membership test in list
    found = n - 1 in my_list
    
    print(f"  O(n) operations complete - processed {n} items")

def demonstrate_on2(n):
    """O(n²) - Quadratic time"""
    my_list = list(range(n))
    count = 0
    
    # O(n²) - nested loops
    for i in my_list:
        for j in my_list:
            count += 1
    
    print(f"  O(n²) operations complete - {count} iterations for {n} items")

# Demonstrate complexities
demonstrate_o1()
demonstrate_on(100)
demonstrate_on2(10)  # Smaller n for O(n²) demo


# ----------------------------------------------------------------------------
# 1.4 Python Data Structure Time Complexities
# ----------------------------------------------------------------------------
print("\n--- 1.4 Python Data Structure Time Complexities ---")
print("""
LIST OPERATIONS:
- Access by index: O(1)
- Append: O(1) amortized
- Insert at position i: O(n)
- Remove by value: O(n)
- Remove by index: O(n)
- Membership test (in): O(n)
- Sort: O(n log n)
- Slice: O(k) where k is slice length

DICTIONARY OPERATIONS:
- Access by key: O(1) average, O(n) worst case
- Insert: O(1) average, O(n) worst case
- Delete: O(1) average, O(n) worst case
- Membership test (in): O(1) average, O(n) worst case
- Iteration: O(n)

SET OPERATIONS:
- Add: O(1) average, O(n) worst case
- Remove: O(1) average, O(n) worst case
- Membership test (in): O(1) average, O(n) worst case
- Union: O(n + m)
- Intersection: O(min(n, m))
- Difference: O(n)

STRING OPERATIONS:
- Access by index: O(1)
- Concatenation: O(n + m) - creates new string
- Membership test (in): O(n)
- Slice: O(k) where k is slice length
- Find/replace: O(n)
""")


# ============================================================================
# 2. SPACE COMPLEXITY - MEMORY USAGE
# ============================================================================

print("\n" + "=" * 70)
print("2. SPACE COMPLEXITY - MEMORY USAGE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Understanding Space Complexity
# ----------------------------------------------------------------------------
print("\n--- 2.1 Understanding Space Complexity ---")
print("""
SPACE COMPLEXITY describes memory usage:
- O(1): Constant space - fixed memory regardless of input
- O(n): Linear space - memory grows with input
- O(n²): Quadratic space - memory grows quadratically

AUXILIARY SPACE: Extra space used by algorithm (excluding input)
TOTAL SPACE: Input space + auxiliary space
""")


# ----------------------------------------------------------------------------
# 2.2 Memory Usage Examples
# ----------------------------------------------------------------------------
print("\n--- 2.2 Memory Usage Examples ---")

def get_size(obj):
    """Get size of object in bytes"""
    return sys.getsizeof(obj)

# O(1) Space
def constant_space_example(n):
    """O(1) - constant space"""
    total = 0
    for i in range(n):
        total += i
    return total  # Only uses fixed variables

# O(n) Space
def linear_space_example(n):
    """O(n) - linear space"""
    result = []
    for i in range(n):
        result.append(i)
    return result  # Creates list of size n

# O(n²) Space
def quadratic_space_example(n):
    """O(n²) - quadratic space"""
    matrix = []
    for i in range(n):
        row = [0] * n
        matrix.append(row)
    return matrix  # Creates n×n matrix

print("Memory usage examples:")
print(f"  Constant space function: ~{get_size(constant_space_example(1000))} bytes")
print(f"  Linear space (n=100): ~{get_size(linear_space_example(100))} bytes")
print(f"  Quadratic space (n=10): ~{get_size(quadratic_space_example(10))} bytes")


# ----------------------------------------------------------------------------
# 2.3 Memory-Efficient vs Memory-Intensive Patterns
# ----------------------------------------------------------------------------
print("\n--- 2.3 Memory-Efficient vs Memory-Intensive Patterns ---")

# Memory-Efficient: Generators
def memory_efficient_generator(n):
    """Generator - O(1) space"""
    for i in range(n):
        yield i ** 2

# Memory-Intensive: List Comprehension
def memory_intensive_list(n):
    """List comprehension - O(n) space"""
    return [i ** 2 for i in range(n)]

print("\nMemory-efficient (generator):")
gen = memory_efficient_generator(1000)
print(f"  Generator object size: {get_size(gen)} bytes")
print("  Processes one item at a time - O(1) space")

print("\nMemory-intensive (list):")
lst = memory_intensive_list(1000)
print(f"  List size: {get_size(lst)} bytes")
print("  Stores all items in memory - O(n) space")


# ----------------------------------------------------------------------------
# 2.4 Data Structure Memory Overhead
# ----------------------------------------------------------------------------
print("\n--- 2.4 Data Structure Memory Overhead ---")

n = 1000
my_list = list(range(n))
my_tuple = tuple(range(n))
my_set = set(range(n))
my_dict = {i: i for i in range(n)}

print("Memory usage for 1000 elements:")
print(f"  List: ~{get_size(my_list)} bytes")
print(f"  Tuple: ~{get_size(my_tuple)} bytes")
print(f"  Set: ~{get_size(my_set)} bytes")
print(f"  Dict: ~{get_size(my_dict)} bytes")
print("\nNote: Sets and dicts have hash table overhead (~3x elements)")


# ============================================================================
# 3. LIST VS SET - WHEN TO USE WHICH
# ============================================================================

print("\n" + "=" * 70)
print("3. LIST VS SET - WHEN TO USE WHICH")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Key Differences
# ----------------------------------------------------------------------------
print("\n--- 3.1 Key Differences ---")
print("""
LIST:
- Ordered collection (maintains insertion order)
- Allows duplicates
- Mutable (can modify elements)
- Access by index: O(1)
- Membership test: O(n)
- Use when: Need order, duplicates, or index access

SET:
- Unordered collection (no guaranteed order)
- No duplicates (unique elements only)
- Mutable (can add/remove)
- No index access
- Membership test: O(1) average
- Use when: Need fast membership testing or unique elements
""")


# ----------------------------------------------------------------------------
# 3.2 Membership Testing Comparison
# ----------------------------------------------------------------------------
print("\n--- 3.2 Membership Testing Comparison ---")

def compare_membership_test():
    """Compare list vs set membership testing"""
    n = 10000
    
    # Create list and set with same elements
    my_list = list(range(n))
    my_set = set(range(n))
    
    # Test membership in list (O(n))
    start = time.time()
    for _ in range(1000):
        _ = n - 1 in my_list
    list_time = time.time() - start
    
    # Test membership in set (O(1))
    start = time.time()
    for _ in range(1000):
        _ = n - 1 in my_set
    set_time = time.time() - start
    
    print(f"Membership test (1000 iterations, n={n}):")
    print(f"  List (O(n)): {list_time:.6f} seconds")
    print(f"  Set (O(1)): {set_time:.6f} seconds")
    print(f"  Set is ~{list_time/set_time:.1f}x faster")

compare_membership_test()


# ----------------------------------------------------------------------------
# 3.3 When to Use List
# ----------------------------------------------------------------------------
print("\n--- 3.3 When to Use List ---")
print("""
USE LIST WHEN:
1. Need to maintain order
   my_list = ['first', 'second', 'third']

2. Need duplicates
   my_list = [1, 2, 2, 3, 3, 3]

3. Need index access
   my_list[0]  # O(1)

4. Need to iterate in order
   for item in my_list:  # Guaranteed order

5. Need slicing
   my_list[1:3]  # Get subset

6. Small dataset where O(n) membership is acceptable
""")


# ----------------------------------------------------------------------------
# 3.4 When to Use Set
# ----------------------------------------------------------------------------
print("\n--- 3.4 When to Use Set ---")
print("""
USE SET WHEN:
1. Need fast membership testing
   if item in my_set:  # O(1) vs O(n) for list

2. Need unique elements
   my_set = {1, 2, 3}  # Automatically removes duplicates

3. Need set operations (union, intersection, difference)
   set1 | set2  # Union
   set1 & set2  # Intersection
   set1 - set2  # Difference

4. Don't care about order
   my_set = {3, 1, 2}  # Order not guaranteed

5. Large dataset where membership testing is frequent
""")


# ----------------------------------------------------------------------------
# 3.5 Practical Examples
# ----------------------------------------------------------------------------
print("\n--- 3.5 Practical Examples ---")

# Example 1: Finding duplicates
def find_duplicates_list(data):
    """Find duplicates using list - O(n²)"""
    duplicates = []
    for i in range(len(data)):
        if data[i] in data[i+1:] and data[i] not in duplicates:
            duplicates.append(data[i])
    return duplicates

def find_duplicates_set(data):
    """Find duplicates using set - O(n)"""
    seen = set()
    duplicates = set()
    for item in data:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)

data = [1, 2, 3, 2, 4, 3, 5, 2]
print(f"Finding duplicates in: {data}")
print(f"  Using list: {find_duplicates_list(data)}")
print(f"  Using set: {find_duplicates_set(data)}")

# Example 2: Removing duplicates
def remove_duplicates_list(data):
    """Remove duplicates using list - O(n²)"""
    result = []
    for item in data:
        if item not in result:
            result.append(item)
    return result

def remove_duplicates_set(data):
    """Remove duplicates using set - O(n)"""
    return list(set(data))  # Note: loses order

data = [1, 2, 3, 2, 4, 3, 5]
print(f"\nRemoving duplicates from: {data}")
print(f"  Using list (preserves order): {remove_duplicates_list(data)}")
print(f"  Using set (faster, loses order): {remove_duplicates_set(data)}")

# Example 3: Intersection
def intersection_list(list1, list2):
    """Find intersection using list - O(n*m)"""
    return [x for x in list1 if x in list2]

def intersection_set(set1, set2):
    """Find intersection using set - O(min(n, m))"""
    return set1 & set2

list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
print(f"\nIntersection of {list1} and {list2}:")
print(f"  Using list: {intersection_list(list1, list2)}")
print(f"  Using set: {list(intersection_set(set(list1), set(list2)))}")


# ============================================================================
# 4. COMPREHENSIONS VS LOOPS - PERFORMANCE TRADE-OFFS
# ============================================================================

print("\n" + "=" * 70)
print("4. COMPREHENSIONS VS LOOPS - PERFORMANCE TRADE-OFFS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 List Comprehension vs Loop
# ----------------------------------------------------------------------------
print("\n--- 4.1 List Comprehension vs Loop ---")

def squares_loop(n):
    """Using traditional loop"""
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

def squares_comprehension(n):
    """Using list comprehension"""
    return [i ** 2 for i in range(n)]

# Performance comparison
n = 100000
start = time.time()
_ = squares_loop(n)
loop_time = time.time() - start

start = time.time()
_ = squares_comprehension(n)
comp_time = time.time() - start

print(f"Creating list of squares (n={n}):")
print(f"  Loop: {loop_time:.6f} seconds")
print(f"  Comprehension: {comp_time:.6f} seconds")
print(f"  Comprehension is ~{loop_time/comp_time:.2f}x faster")


# ----------------------------------------------------------------------------
# 4.2 Why Comprehensions Are Faster
# ----------------------------------------------------------------------------
print("\n--- 4.2 Why Comprehensions Are Faster ---")
print("""
REASONS COMPREHENSIONS ARE FASTER:
1. Optimized bytecode - Python optimizes comprehensions
2. No function call overhead - append() is called in C
3. Less overhead - fewer Python-level operations
4. Better memory allocation - pre-allocates list size when possible

WHEN TO USE LOOPS:
1. Complex logic that's hard to read in comprehension
2. Need side effects (printing, file I/O)
3. Need break/continue statements
4. Multiple statements needed per iteration
""")


# ----------------------------------------------------------------------------
# 4.3 Generator Expression vs List Comprehension
# ----------------------------------------------------------------------------
print("\n--- 4.3 Generator Expression vs List Comprehension ---")

def sum_squares_list_comp(n):
    """Using list comprehension - O(n) space"""
    return sum([i ** 2 for i in range(n)])

def sum_squares_gen_expr(n):
    """Using generator expression - O(1) space"""
    return sum(i ** 2 for i in range(n))

# Memory comparison
n = 1000000
print(f"Sum of squares (n={n}):")
print(f"  List comprehension memory: ~{get_size([i ** 2 for i in range(1000)])} bytes (for 1000 items)")
print(f"  Generator expression memory: ~{get_size((i ** 2 for i in range(1000)))} bytes (constant)")

# Performance comparison
start = time.time()
_ = sum_squares_list_comp(n)
list_time = time.time() - start

start = time.time()
_ = sum_squares_gen_expr(n)
gen_time = time.time() - start

print(f"\nPerformance:")
print(f"  List comprehension: {list_time:.6f} seconds")
print(f"  Generator expression: {gen_time:.6f} seconds")
print(f"  Generator is ~{list_time/gen_time:.2f}x faster (and uses less memory)")


# ----------------------------------------------------------------------------
# 4.4 Dictionary Comprehension vs Loop
# ----------------------------------------------------------------------------
print("\n--- 4.4 Dictionary Comprehension vs Loop ---")

def dict_loop(keys, values):
    """Create dict using loop"""
    result = {}
    for k, v in zip(keys, values):
        result[k] = v
    return result

def dict_comprehension(keys, values):
    """Create dict using comprehension"""
    return {k: v for k, v in zip(keys, values)}

keys = list(range(10000))
values = [i ** 2 for i in keys]

start = time.time()
_ = dict_loop(keys, values)
loop_time = time.time() - start

start = time.time()
_ = dict_comprehension(keys, values)
comp_time = time.time() - start

print(f"Creating dictionary (n=10000):")
print(f"  Loop: {loop_time:.6f} seconds")
print(f"  Comprehension: {comp_time:.6f} seconds")
print(f"  Comprehension is ~{loop_time/comp_time:.2f}x faster")


# ----------------------------------------------------------------------------
# 4.5 When Comprehensions Are NOT Better
# ----------------------------------------------------------------------------
print("\n--- 4.5 When Comprehensions Are NOT Better ---")
print("""
USE LOOPS WHEN:

1. Complex logic that hurts readability:
   # Bad - hard to read
   result = [x if x > 0 else (y if y > 0 else z) for x, y, z in data]
   
   # Good - clear logic
   result = []
   for x, y, z in data:
       if x > 0:
           result.append(x)
       elif y > 0:
           result.append(y)
       else:
           result.append(z)

2. Need side effects:
   # Can't do this in comprehension
   for item in data:
       print(item)
       process(item)
       log(item)

3. Need break/continue:
   # Can't break/continue in comprehension
   result = []
   for item in data:
       if condition(item):
           break
       result.append(process(item))

4. Multiple statements per iteration:
   # Comprehensions are single-expression only
   result = []
   for item in data:
       processed = transform(item)
       validated = validate(processed)
       result.append(validated)
""")


# ----------------------------------------------------------------------------
# 4.6 Nested Comprehensions Performance
# ----------------------------------------------------------------------------
print("\n--- 4.6 Nested Comprehensions Performance ---")

def nested_loop(n):
    """Nested loop"""
    result = []
    for i in range(n):
        for j in range(n):
            result.append(i * j)
    return result

def nested_comprehension(n):
    """Nested comprehension"""
    return [i * j for i in range(n) for j in range(n)]

n = 100
start = time.time()
_ = nested_loop(n)
loop_time = time.time() - start

start = time.time()
_ = nested_comprehension(n)
comp_time = time.time() - start

print(f"Nested loops (n={n}, total iterations={n*n}):")
print(f"  Loop: {loop_time:.6f} seconds")
print(f"  Comprehension: {comp_time:.6f} seconds")
print(f"  Comprehension is ~{loop_time/comp_time:.2f}x faster")
print("\nNote: Both are O(n²) time and space, but comprehension is faster")


# ============================================================================
# 5. PRACTICAL PERFORMANCE TIPS
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICAL PERFORMANCE TIPS")
print("=" * 70)

print("""
1. MEMBERSHIP TESTING:
   - Use set for frequent membership tests
   - Convert list to set if testing many times
   - Example: if item in my_set (not my_list)

2. AVOID PREMATURE OPTIMIZATION:
   - Write readable code first
   - Profile before optimizing
   - Optimize bottlenecks only

3. USE BUILT-IN FUNCTIONS:
   - Built-ins are implemented in C (faster)
   - Example: sum(), max(), min(), sorted()

4. AVOID UNNECESSARY OPERATIONS:
   - Don't create intermediate lists if not needed
   - Use generator expressions for large data
   - Example: sum(x**2 for x in range(n)) not sum([x**2 for x in range(n)])

5. STRING OPERATIONS:
   - Use join() instead of += in loops
   - Use f-strings for formatting
   - Example: ''.join(words) not result += word

6. LIST OPERATIONS:
   - Use append() instead of insert(0, x)
   - Use extend() for multiple items
   - Consider deque for frequent insertions at both ends

7. DICTIONARY OPERATIONS:
   - Use dict.get() with default instead of try/except
   - Use dict.setdefault() or defaultdict
   - Example: value = my_dict.get(key, default)

8. MEMORY MANAGEMENT:
   - Use generators for large datasets
   - Delete large objects when done: del large_object
   - Process data in chunks when possible
""")


# ============================================================================
# 6. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Choose the right data structure
print("\n--- Exercise 1: Choose the Right Data Structure ---")
print("""
Problem: You need to check if a value exists in a collection frequently.
Which is better: list or set?

Answer: SET
- Set membership: O(1) average
- List membership: O(n)
- Set is much faster for frequent lookups
""")

# Exercise 2: Optimize membership testing
print("\n--- Exercise 2: Optimize Membership Testing ---")
def check_membership_inefficient(data, targets):
    """Inefficient - O(n*m)"""
    result = []
    for target in targets:
        if target in data:  # O(n) for list
            result.append(target)
    return result

def check_membership_efficient(data, targets):
    """Efficient - O(n+m)"""
    data_set = set(data)  # O(n) conversion
    return [target for target in targets if target in data_set]  # O(1) lookup

data = list(range(10000))
targets = [9999, 5000, 1, 9998]
print(f"Finding {targets} in data:")
print(f"  Inefficient (list): {check_membership_inefficient(data, targets)}")
print(f"  Efficient (set): {check_membership_efficient(data, targets)}")

# Exercise 3: Choose comprehension vs loop
print("\n--- Exercise 3: Choose Comprehension vs Loop ---")
print("""
Problem: Create list of squares for even numbers only.

Comprehension (better):
  result = [x**2 for x in range(n) if x % 2 == 0]

Loop (acceptable but slower):
  result = []
  for x in range(n):
      if x % 2 == 0:
          result.append(x**2)
""")

# Exercise 4: Memory-efficient processing
print("\n--- Exercise 4: Memory-Efficient Processing ---")
print("""
Problem: Process large dataset without loading all into memory.

Use generator:
  def process_large_file(filename):
      with open(filename) as f:
          for line in f:
              yield process(line)

Not list:
  def process_large_file_bad(filename):
      with open(filename) as f:
          return [process(line) for line in f]  # Loads all into memory
""")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. TIME COMPLEXITY:
   - Know common operations: O(1), O(log n), O(n), O(n log n), O(n²)
   - Understand when operations are O(1) vs O(n)
   - Be able to analyze your code's complexity

2. SPACE COMPLEXITY:
   - Consider memory usage, not just time
   - Generators save memory for large datasets
   - Be aware of data structure overhead

3. LIST VS SET:
   - List: Ordered, duplicates, index access, O(n) membership
   - Set: Unordered, unique, O(1) membership, set operations
   - Choose based on use case (membership testing frequency)

4. COMPREHENSIONS VS LOOPS:
   - Comprehensions are faster and more Pythonic
   - Use for simple transformations and filtering
   - Use loops for complex logic or side effects
   - Generator expressions save memory

5. PERFORMANCE BEST PRACTICES:
   - Use appropriate data structures
   - Avoid premature optimization
   - Profile before optimizing
   - Use built-in functions when possible
   - Consider memory for large datasets

6. INTERVIEW TIPS:
   - Mention time/space complexity when discussing solutions
   - Explain trade-offs (e.g., list vs set)
   - Consider both time and space when optimizing
   - Use comprehensions when appropriate
   - Choose data structures based on operations needed
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Performance Considerations Guide Ready!")
    print("=" * 70)
