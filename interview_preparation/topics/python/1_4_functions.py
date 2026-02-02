"""
Python Functions - Interview Preparation
Topic 1.4: Functions

This module covers:
- Function Definition: def, parameters, return
- Arguments: Positional, keyword, default parameters
- Variable Arguments: *args, **kwargs
- Lambda Functions: Anonymous functions, map, filter, reduce
- Scope: Local, global, nonlocal, LEGB rule
- Function Annotations: Type hints
- Recursion: Base cases, recursive calls
"""

# ============================================================================
# 1. FUNCTION DEFINITION
# ============================================================================

print("=" * 70)
print("1. FUNCTION DEFINITION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Basic Function Definition
# ----------------------------------------------------------------------------
print("\n--- Basic Function Definition ---")

def greet():
    """Simple function with no parameters."""
    print("Hello, World!")

greet()

# Function with parameters
def greet_person(name):
    """Function with one parameter."""
    print(f"Hello, {name}!")

greet_person("Alice")

# Function with return statement
def add(a, b):
    """Function that returns a value."""
    return a + b

result = add(3, 5)
print(f"3 + 5 = {result}")

# Function without explicit return (returns None)
def print_sum(a, b):
    """Function without return statement."""
    print(f"{a} + {b} = {a + b}")

result = print_sum(2, 3)
print(f"Return value: {result}")  # None


# ----------------------------------------------------------------------------
# 1.2 Multiple Parameters
# ----------------------------------------------------------------------------
print("\n--- Multiple Parameters ---")

def calculate_area(length, width):
    """Calculate area of rectangle."""
    return length * width

area = calculate_area(5, 3)
print(f"Area of 5x3 rectangle: {area}")

# Function with multiple return values (returns tuple)
def get_name_and_age():
    """Function returning multiple values."""
    return "Alice", 25

name, age = get_name_and_age()
print(f"Name: {name}, Age: {age}")

# Unpacking return values
def divide_with_remainder(a, b):
    """Return quotient and remainder."""
    quotient = a // b
    remainder = a % b
    return quotient, remainder

q, r = divide_with_remainder(17, 5)
print(f"17 ÷ 5 = {q} remainder {r}")


# ----------------------------------------------------------------------------
# 1.3 Docstrings
# ----------------------------------------------------------------------------
print("\n--- Docstrings ---")

def power(base, exponent):
    """
    Calculate base raised to the power of exponent.
    
    Args:
        base (int or float): The base number
        exponent (int or float): The exponent
    
    Returns:
        int or float: base raised to the power of exponent
    """
    return base ** exponent

print(f"2^3 = {power(2, 3)}")
print(f"Function docstring: {power.__doc__}")


# ============================================================================
# 2. ARGUMENTS
# ============================================================================

print("\n" + "=" * 70)
print("2. ARGUMENTS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Positional Arguments
# ----------------------------------------------------------------------------
print("\n--- Positional Arguments ---")

def subtract(a, b):
    """Subtract b from a."""
    return a - b

# Positional arguments (order matters)
result = subtract(10, 3)
print(f"10 - 3 = {result}")

# Wrong order gives wrong result
result = subtract(3, 10)
print(f"3 - 10 = {result}")  # -7 (not 7)


# ----------------------------------------------------------------------------
# 2.2 Keyword Arguments
# ----------------------------------------------------------------------------
print("\n--- Keyword Arguments ---")

def greet_full_name(first_name, last_name):
    """Greet with full name."""
    print(f"Hello, {first_name} {last_name}!")

# Using keyword arguments (order doesn't matter)
greet_full_name(first_name="John", last_name="Doe")
greet_full_name(last_name="Smith", first_name="Jane")

# Mixing positional and keyword arguments
def create_profile(name, age, city, country):
    """Create a profile."""
    print(f"{name}, {age}, from {city}, {country}")

# Positional first, then keyword
create_profile("Alice", 25, city="New York", country="USA")

# All keyword
create_profile(name="Bob", age=30, city="London", country="UK")


# ----------------------------------------------------------------------------
# 2.3 Default Parameters
# ----------------------------------------------------------------------------
print("\n--- Default Parameters ---")

def greet_with_title(name, title="Mr."):
    """Greet with optional title."""
    print(f"Hello, {title} {name}!")

# Using default parameter
greet_with_title("Smith")  # Uses default "Mr."

# Overriding default
greet_with_title("Smith", "Dr.")

# Multiple default parameters
def create_email(to, subject="No Subject", body=""):
    """Create an email."""
    print(f"To: {to}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")

create_email("alice@example.com")
create_email("bob@example.com", "Meeting")
create_email("charlie@example.com", "Hello", "How are you?")

# Important: Default parameters are evaluated once
def add_item(item, items=[]):  # WARNING: Mutable default argument
    """Add item to list (has a bug!)."""
    items.append(item)
    return items

# This is problematic!
list1 = add_item("apple")
list2 = add_item("banana")
print(f"list1: {list1}")  # ['apple', 'banana'] - unexpected!
print(f"list2: {list2}")  # ['apple', 'banana'] - unexpected!

# Correct way: Use None as default
def add_item_correct(item, items=None):
    """Add item to list (correct version)."""
    if items is None:
        items = []
    items.append(item)
    return items

list1 = add_item_correct("apple")
list2 = add_item_correct("banana")
print(f"list1: {list1}")  # ['apple']
print(f"list2: {list2}")  # ['banana']


# ----------------------------------------------------------------------------
# 2.4 Parameter Order Rules
# ----------------------------------------------------------------------------
print("\n--- Parameter Order Rules ---")

def example_function(pos1, pos2, /, pos_or_kw, *, kw1, kw2):
    """
    Function demonstrating parameter order rules.
    
    / : Separates positional-only from positional-or-keyword
    * : Separates positional-or-keyword from keyword-only
    """
    print(f"pos1={pos1}, pos2={pos2}, pos_or_kw={pos_or_kw}, kw1={kw1}, kw2={kw2}")

# pos1 and pos2 must be positional
# pos_or_kw can be positional or keyword
# kw1 and kw2 must be keyword
example_function(1, 2, 3, kw1=4, kw2=5)
example_function(1, 2, pos_or_kw=3, kw1=4, kw2=5)


# ============================================================================
# 3. VARIABLE ARGUMENTS (*args and **kwargs)
# ============================================================================

print("\n" + "=" * 70)
print("3. VARIABLE ARGUMENTS (*args and **kwargs)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 *args - Variable Positional Arguments
# ----------------------------------------------------------------------------
print("\n--- *args - Variable Positional Arguments ---")

def sum_all(*args):
    """Sum all arguments."""
    total = 0
    for num in args:
        total += num
    return total

print(f"sum_all(1, 2, 3) = {sum_all(1, 2, 3)}")
print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")

# *args is a tuple
def print_args(*args):
    """Print all arguments."""
    print(f"Type of args: {type(args)}")
    print(f"Args: {args}")

print_args(1, 2, 3, "hello", True)

# Mixing regular parameters with *args
def greet_multiple(greeting, *names):
    """Greet multiple people."""
    for name in names:
        print(f"{greeting}, {name}!")

greet_multiple("Hello", "Alice", "Bob", "Charlie")

# Unpacking with *
def multiply(a, b, c):
    """Multiply three numbers."""
    return a * b * c

numbers = [2, 3, 4]
result = multiply(*numbers)  # Unpacks list as arguments
print(f"multiply(*[2, 3, 4]) = {result}")


# ----------------------------------------------------------------------------
# 3.2 **kwargs - Variable Keyword Arguments
# ----------------------------------------------------------------------------
print("\n--- **kwargs - Variable Keyword Arguments ---")

def print_info(**kwargs):
    """Print all keyword arguments."""
    print(f"Type of kwargs: {type(kwargs)}")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="New York")

# **kwargs is a dictionary
def create_profile(**kwargs):
    """Create profile from keyword arguments."""
    profile = {}
    for key, value in kwargs.items():
        profile[key] = value
    return profile

profile = create_profile(name="Bob", age=30, occupation="Engineer")
print(f"Profile: {profile}")

# Unpacking dictionary with **
def introduce(name, age, city):
    """Introduce a person."""
    print(f"{name}, {age} years old, from {city}")

person = {"name": "Charlie", "age": 35, "city": "London"}
introduce(**person)  # Unpacks dictionary as keyword arguments


# ----------------------------------------------------------------------------
# 3.3 Combining *args and **kwargs
# ----------------------------------------------------------------------------
print("\n--- Combining *args and **kwargs ---")

def flexible_function(required, *args, **kwargs):
    """
    Function with required, *args, and **kwargs.
    
    Order must be: required, *args, **kwargs
    """
    print(f"Required: {required}")
    print(f"*args: {args}")
    print(f"**kwargs: {kwargs}")

flexible_function("first", 1, 2, 3, name="Alice", age=25)

# Common pattern: Wrapper functions
def wrapper_function(*args, **kwargs):
    """Wrapper that passes arguments to another function."""
    print("Wrapper: Preparing to call function")
    # In real code, you'd call another function here
    print(f"Args: {args}, Kwargs: {kwargs}")


# ============================================================================
# 4. LAMBDA FUNCTIONS
# ============================================================================

print("\n" + "=" * 70)
print("4. LAMBDA FUNCTIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Basic Lambda Functions
# ----------------------------------------------------------------------------
print("\n--- Basic Lambda Functions ---")

# Regular function
def square(x):
    return x ** 2

# Equivalent lambda function
square_lambda = lambda x: x ** 2

print(f"square(5) = {square(5)}")
print(f"square_lambda(5) = {square_lambda(5)}")

# Lambda with multiple parameters
add_lambda = lambda a, b: a + b
print(f"add_lambda(3, 4) = {add_lambda(3, 4)}")

# Lambda with no parameters
get_hello = lambda: "Hello, World!"
print(get_hello())


# ----------------------------------------------------------------------------
# 4.2 Lambda with map()
# ----------------------------------------------------------------------------
print("\n--- Lambda with map() ---")

# map(function, iterable) - applies function to each element
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(f"Squared {numbers}: {squared}")

# Multiple iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
sums = list(map(lambda x, y: x + y, list1, list2))
print(f"Sum of {list1} and {list2}: {sums}")

# With regular function
def double(x):
    return x * 2

doubled = list(map(double, numbers))
print(f"Doubled {numbers}: {doubled}")


# ----------------------------------------------------------------------------
# 4.3 Lambda with filter()
# ----------------------------------------------------------------------------
print("\n--- Lambda with filter() ---")

# filter(function, iterable) - filters elements where function returns True
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers from {numbers}: {evens}")

# Filter strings
words = ["apple", "banana", "cherry", "date", "elderberry"]
long_words = list(filter(lambda word: len(word) > 5, words))
print(f"Words longer than 5 chars: {long_words}")

# With regular function
def is_positive(x):
    return x > 0

numbers = [-2, -1, 0, 1, 2, 3]
positives = list(filter(is_positive, numbers))
print(f"Positive numbers: {positives}")


# ----------------------------------------------------------------------------
# 4.4 Lambda with reduce()
# ----------------------------------------------------------------------------
print("\n--- Lambda with reduce() ---")

from functools import reduce

# reduce(function, iterable) - reduces iterable to single value
numbers = [1, 2, 3, 4, 5]
sum_all = reduce(lambda x, y: x + y, numbers)
print(f"Sum of {numbers}: {sum_all}")

# Find maximum
max_value = reduce(lambda x, y: x if x > y else y, numbers)
print(f"Max of {numbers}: {max_value}")

# Multiply all
product = reduce(lambda x, y: x * y, numbers)
print(f"Product of {numbers}: {product}")


# ----------------------------------------------------------------------------
# 4.5 Lambda in sorted() and list.sort()
# ----------------------------------------------------------------------------
print("\n--- Lambda in sorted() and list.sort() ---")

# Sort by length
words = ["apple", "pie", "banana", "cat"]
sorted_by_length = sorted(words, key=lambda x: len(x))
print(f"Sorted by length: {sorted_by_length}")

# Sort by second character
sorted_by_second = sorted(words, key=lambda x: x[1] if len(x) > 1 else '')
print(f"Sorted by second char: {sorted_by_second}")

# Sort list of tuples
students = [("Alice", 25), ("Bob", 20), ("Charlie", 22)]
sorted_by_age = sorted(students, key=lambda x: x[1])
print(f"Sorted by age: {sorted_by_age}")


# ----------------------------------------------------------------------------
# 4.6 When to Use Lambda
# ----------------------------------------------------------------------------
print("\n--- When to Use Lambda ---")

# Good: Simple one-line functions used once
numbers = [1, 2, 3, 4, 5]
result = list(map(lambda x: x * 2, numbers))

# Better: For complex logic, use regular function
def complex_calculation(x):
    """Complex calculation that needs explanation."""
    if x < 0:
        return 0
    return x ** 2 + 2 * x + 1

result = list(map(complex_calculation, numbers))


# ============================================================================
# 5. SCOPE
# ============================================================================

print("\n" + "=" * 70)
print("5. SCOPE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Local Scope
# ----------------------------------------------------------------------------
print("\n--- Local Scope ---")

def local_example():
    """Function with local variable."""
    local_var = "I'm local"
    print(f"Inside function: {local_var}")

local_example()
# print(local_var)  # Error: NameError - local_var is not defined


# ----------------------------------------------------------------------------
# 5.2 Global Scope
# ----------------------------------------------------------------------------
print("\n--- Global Scope ---")

global_var = "I'm global"

def read_global():
    """Function reading global variable."""
    print(f"Reading global: {global_var}")

read_global()

# Modifying global variable (requires global keyword)
counter = 0

def increment_counter():
    """Increment global counter."""
    global counter
    counter += 1
    print(f"Counter: {counter}")

increment_counter()
increment_counter()
print(f"Final counter: {counter}")


# ----------------------------------------------------------------------------
# 5.3 Enclosing Scope (Nonlocal)
# ----------------------------------------------------------------------------
print("\n--- Enclosing Scope (Nonlocal) ---")

def outer_function():
    """Outer function with enclosing scope."""
    outer_var = "I'm in outer function"
    
    def inner_function():
        """Inner function accessing enclosing scope."""
        print(f"Accessing outer: {outer_var}")
    
    inner_function()

outer_function()

# Modifying enclosing scope variable
def outer_with_counter():
    """Outer function with counter."""
    count = 0
    
    def increment():
        """Increment count in enclosing scope."""
        nonlocal count
        count += 1
        return count
    
    return increment

counter_func = outer_with_counter()
print(f"Count: {counter_func()}")
print(f"Count: {counter_func()}")
print(f"Count: {counter_func()}")


# ----------------------------------------------------------------------------
# 5.4 LEGB Rule
# ----------------------------------------------------------------------------
print("\n--- LEGB Rule ---")

# LEGB: Local -> Enclosing -> Global -> Built-in

# Built-in scope (Python's built-in functions)
print(len([1, 2, 3]))  # len is in built-in scope

# Global scope
x = "global"

def outer():
    """Outer function."""
    x = "enclosing"
    
    def inner():
        """Inner function."""
        x = "local"
        print(f"x in inner: {x}")  # Local
    
    inner()
    print(f"x in outer: {x}")  # Enclosing

outer()
print(f"x in global: {x}")  # Global

# Accessing different scopes
def demonstrate_legb():
    """Demonstrate LEGB lookup."""
    # Local
    x = "local"
    
    def inner():
        # Will find x in enclosing scope
        print(f"Enclosing x: {x}")
    
    inner()

demonstrate_legb()


# ----------------------------------------------------------------------------
# 5.5 Scope Examples
# ----------------------------------------------------------------------------
print("\n--- Scope Examples ---")

# Example 1: Variable shadowing
x = 10

def function1():
    x = 20  # Local x shadows global x
    print(f"Local x: {x}")

function1()
print(f"Global x: {x}")

# Example 2: Modifying mutable objects
my_list = [1, 2, 3]

def modify_list():
    """Modify list without global keyword."""
    my_list.append(4)  # Can modify, but not reassign
    print(f"Modified list: {my_list}")

modify_list()
print(f"List after function: {my_list}")

# Example 3: Reassignment requires global
my_list = [1, 2, 3]

def reassign_list():
    """Reassign list requires global."""
    global my_list
    my_list = [4, 5, 6]
    print(f"Reassigned list: {my_list}")

reassign_list()
print(f"List after reassignment: {my_list}")


# ============================================================================
# 6. FUNCTION ANNOTATIONS (TYPE HINTS)
# ============================================================================

print("\n" + "=" * 70)
print("6. FUNCTION ANNOTATIONS (TYPE HINTS)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 6.1 Basic Type Hints
# ----------------------------------------------------------------------------
print("\n--- Basic Type Hints ---")

def add_annotated(a: int, b: int) -> int:
    """Add two integers with type hints."""
    return a + b

result = add_annotated(3, 5)
print(f"add_annotated(3, 5) = {result}")

# Type hints don't enforce types (Python is still dynamically typed)
result = add_annotated(3.5, 2.5)  # Works, but type checker would warn
print(f"add_annotated(3.5, 2.5) = {result}")

# Accessing annotations
print(f"Function annotations: {add_annotated.__annotations__}")


# ----------------------------------------------------------------------------
# 6.2 Multiple Types
# ----------------------------------------------------------------------------
print("\n--- Multiple Types ---")

from typing import Union, List, Dict, Optional

def process_number(num: Union[int, float]) -> float:
    """Process number that can be int or float."""
    return float(num)

print(f"process_number(5) = {process_number(5)}")
print(f"process_number(3.14) = {process_number(3.14)}")

# Optional (Union with None)
def find_item(items: List[str], target: str) -> Optional[int]:
    """Find index of item, return None if not found."""
    try:
        return items.index(target)
    except ValueError:
        return None

fruits = ["apple", "banana", "cherry"]
print(f"Index of 'banana': {find_item(fruits, 'banana')}")
print(f"Index of 'orange': {find_item(fruits, 'orange')}")


# ----------------------------------------------------------------------------
# 6.3 Complex Type Hints
# ----------------------------------------------------------------------------
print("\n--- Complex Type Hints ---")

def process_data(
    numbers: List[int],
    metadata: Dict[str, str],
    optional_value: Optional[int] = None
) -> Dict[str, Union[int, List[int]]]:
    """Process data with complex type hints."""
    result = {
        "sum": sum(numbers),
        "count": len(numbers),
        "numbers": numbers
    }
    if optional_value:
        result["optional"] = optional_value
    return result

data = process_data([1, 2, 3, 4], {"key": "value"}, optional_value=10)
print(f"Processed data: {data}")


# ----------------------------------------------------------------------------
# 6.4 Return Type Annotations
# ----------------------------------------------------------------------------
print("\n--- Return Type Annotations ---")

def get_name() -> str:
    """Return name."""
    return "Alice"

def get_age() -> int:
    """Return age."""
    return 25

def get_info() -> tuple[str, int]:
    """Return name and age."""
    return "Bob", 30

name, age = get_info()
print(f"Name: {name}, Age: {age}")


# ============================================================================
# 7. RECURSION
# ============================================================================

print("\n" + "=" * 70)
print("7. RECURSION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 7.1 Basic Recursion
# ----------------------------------------------------------------------------
print("\n--- Basic Recursion ---")

def factorial(n: int) -> int:
    """
    Calculate factorial using recursion.
    
    Base case: n == 0 or n == 1
    Recursive case: n * factorial(n - 1)
    """
    # Base case
    if n == 0 or n == 1:
        return 1
    # Recursive case
    return n * factorial(n - 1)

print(f"factorial(5) = {factorial(5)}")
print(f"factorial(0) = {factorial(0)}")


# ----------------------------------------------------------------------------
# 7.2 Fibonacci Sequence
# ----------------------------------------------------------------------------
print("\n--- Fibonacci Sequence ---")

def fibonacci(n: int) -> int:
    """
    Calculate nth Fibonacci number.
    
    Base cases: n == 0 returns 0, n == 1 returns 1
    Recursive case: fibonacci(n-1) + fibonacci(n-2)
    """
    # Base cases
    if n == 0:
        return 0
    if n == 1:
        return 1
    # Recursive case
    return fibonacci(n - 1) + fibonacci(n - 2)

print("Fibonacci sequence:")
for i in range(10):
    print(f"fibonacci({i}) = {fibonacci(i)}")


# ----------------------------------------------------------------------------
# 7.3 Recursive List Operations
# ----------------------------------------------------------------------------
print("\n--- Recursive List Operations ---")

def sum_list(numbers: List[int]) -> int:
    """Sum list recursively."""
    # Base case
    if not numbers:
        return 0
    # Recursive case
    return numbers[0] + sum_list(numbers[1:])

numbers = [1, 2, 3, 4, 5]
print(f"sum_list({numbers}) = {sum_list(numbers)}")

def find_max(numbers: List[int]) -> int:
    """Find maximum in list recursively."""
    # Base case
    if len(numbers) == 1:
        return numbers[0]
    # Recursive case
    rest_max = find_max(numbers[1:])
    return numbers[0] if numbers[0] > rest_max else rest_max

print(f"find_max({numbers}) = {find_max(numbers)}")


# ----------------------------------------------------------------------------
# 7.4 Recursive String Operations
# ----------------------------------------------------------------------------
print("\n--- Recursive String Operations ---")

def reverse_string(s: str) -> str:
    """Reverse string recursively."""
    # Base case
    if len(s) <= 1:
        return s
    # Recursive case
    # Take the last character of the string and concatenate it with the reversed result of the remaining string.
    return s[-1] + reverse_string(s[:-1])

text = "Python"
print(f"reverse_string('{text}') = '{reverse_string(text)}'")

def is_palindrome(s: str) -> bool:
    """Check if string is palindrome recursively."""
    # Base cases
    if len(s) <= 1:
        return True
    # Recursive case
    if s[0] != s[-1]:
        return False
    # Call the function recursively with the string excluding the first and last characters.
    return is_palindrome(s[1:-1])

print(f"is_palindrome('racecar') = {is_palindrome('racecar')}")
print(f"is_palindrome('hello') = {is_palindrome('hello')}")


# ----------------------------------------------------------------------------
# 7.5 Tail Recursion vs Head Recursion
# ----------------------------------------------------------------------------
print("\n--- Tail Recursion vs Head Recursion ---")

# Head recursion (operation after recursive call)
def countdown_head(n: int):
    """Countdown using head recursion."""
    if n > 0:
        countdown_head(n - 1)  # Recursive call first
        print(n)  # Operation after

print("Head recursion countdown:")
countdown_head(5)

# Tail recursion (operation before recursive call)
def countdown_tail(n: int):
    """Countdown using tail recursion."""
    if n > 0:
        print(n)  # Operation first
        countdown_tail(n - 1)  # Recursive call after

print("\nTail recursion countdown:")
countdown_tail(5)


# ----------------------------------------------------------------------------
# 7.6 Recursive Tree Traversal
# ----------------------------------------------------------------------------
print("\n--- Recursive Tree Traversal ---")

def calculate_tree_depth(node, depth=0):
    """Calculate depth of tree recursively."""
    if node is None:
        return depth
    # Recursively calculate depth of left and right subtrees
    left_depth = calculate_tree_depth(node.get('left'), depth + 1) if node.get('left') else depth
    right_depth = calculate_tree_depth(node.get('right'), depth + 1) if node.get('right') else depth
    return max(left_depth, right_depth)

# Example tree structure
tree = {
    'value': 1,
    'left': {
        'value': 2,
        'left': {'value': 4},
        'right': {'value': 5}
    },
    'right': {'value': 3}
}

depth = calculate_tree_depth(tree)
print(f"Tree depth: {depth}")


# ----------------------------------------------------------------------------
# 7.7 Recursion vs Iteration
# ----------------------------------------------------------------------------
print("\n--- Recursion vs Iteration ---")

# Recursive factorial
def factorial_recursive(n: int) -> int:
    """Factorial using recursion."""
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# Iterative factorial
def factorial_iterative(n: int) -> int:
    """Factorial using iteration."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(f"factorial_recursive(5) = {factorial_recursive(5)}")
print(f"factorial_iterative(5) = {factorial_iterative(5)}")

# Recursion: More elegant, but can cause stack overflow for large n
# Iteration: More efficient, no stack overflow risk


# ============================================================================
# 8. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("8. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Function with default parameters
print("\n--- Exercise 1: Function with Default Parameters ---")
def create_greeting(name: str, greeting: str = "Hello", punctuation: str = "!") -> str:
    """Create a greeting message."""
    return f"{greeting}, {name}{punctuation}"

print(create_greeting("Alice"))
print(create_greeting("Bob", "Hi"))
print(create_greeting("Charlie", "Hey", "."))


# Exercise 2: Function with *args
print("\n--- Exercise 2: Function with *args ---")
def average(*numbers: float) -> float:
    """Calculate average of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(f"average(1, 2, 3, 4, 5) = {average(1, 2, 3, 4, 5)}")
print(f"average(10, 20, 30) = {average(10, 20, 30)}")


# Exercise 3: Function with **kwargs
print("\n--- Exercise 3: Function with **kwargs ---")
def build_query(**filters: str) -> str:
    """Build SQL-like query from filters."""
    if not filters:
        return "SELECT * FROM table"
    conditions = " AND ".join([f"{key} = '{value}'" for key, value in filters.items()])
    return f"SELECT * FROM table WHERE {conditions}"

query = build_query(name="Alice", age="25", city="New York")
print(f"Query: {query}")


# Exercise 4: Lambda with map and filter
print("\n--- Exercise 4: Lambda with map and filter ---")
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squared_evens = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))
print(f"Squared even numbers: {squared_evens}")


# Exercise 5: Recursive power function
print("\n--- Exercise 5: Recursive Power Function ---")
def power_recursive(base: float, exponent: int) -> float:
    """Calculate base^exponent recursively."""
    if exponent == 0:
        return 1
    if exponent < 0:
        return 1 / power_recursive(base, -exponent)
    return base * power_recursive(base, exponent - 1)

print(f"2^5 = {power_recursive(2, 5)}")
print(f"2^-3 = {power_recursive(2, -3)}")


# Exercise 6: Recursive binary search
print("\n--- Exercise 6: Recursive Binary Search ---")
def binary_search(arr: List[int], target: int, left: int = 0, right: int = None) -> int:
    """Binary search recursively."""
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)

sorted_list = [1, 3, 5, 7, 9, 11, 13, 15]
print(f"Index of 7 in {sorted_list}: {binary_search(sorted_list, 7)}")
print(f"Index of 10 in {sorted_list}: {binary_search(sorted_list, 10)}")


# ============================================================================
# 9. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("9. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between *args and **kwargs?
print("\n--- Q1: *args vs **kwargs ---")
print("""
*args: Variable positional arguments, collected as a tuple
**kwargs: Variable keyword arguments, collected as a dictionary
Both allow functions to accept variable number of arguments
""")

# Q2: What happens with mutable default arguments?
print("\n--- Q2: Mutable Default Arguments ---")
print("""
WARNING: Default arguments are evaluated once when function is defined.
Using mutable defaults (list, dict) can lead to unexpected behavior.
Solution: Use None as default and create new object inside function.
""")

# Q3: Explain LEGB rule
print("\n--- Q3: LEGB Rule ---")
print("""
LEGB: Local -> Enclosing -> Global -> Built-in
Python searches for variables in this order when resolving names.
""")

# Q4: When to use recursion vs iteration?
print("\n--- Q4: Recursion vs Iteration ---")
print("""
Recursion: More elegant for tree/graph problems, divide-and-conquer
Iteration: More efficient, no stack overflow risk, better for simple loops
Python has recursion limit (~1000), so prefer iteration for deep recursion.
""")

# Q5: What are lambda functions used for?
print("\n--- Q5: Lambda Functions ---")
print("""
Lambda: Anonymous functions, typically one-line
Use with: map(), filter(), reduce(), sorted() key parameter
Best for: Simple operations used once, not for complex logic
""")

# Q6: How do type hints work?
print("\n--- Q6: Type Hints ---")
print("""
Type hints: Annotations that document expected types
They don't enforce types (Python is still dynamically typed)
Useful for: Documentation, IDE support, static type checkers (mypy)
""")

# Q7: What is the difference between global and nonlocal?
print("\n--- Q7: global vs nonlocal ---")
print("""
global: Modifies variable in global scope
nonlocal: Modifies variable in enclosing (non-global) scope
Both required when reassigning, not needed for mutating objects
""")


# ============================================================================
# 10. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("10. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. FUNCTION DEFINITION:
   - Use def keyword to define functions
   - Functions without return return None
   - Use docstrings to document functions
   - Functions are first-class objects (can be assigned, passed as arguments)

2. ARGUMENTS:
   - Positional: Order matters
   - Keyword: Order doesn't matter, use name=value
   - Default: Provide default values, evaluated once
   - Never use mutable defaults (use None instead)

3. *args and **kwargs:
   - *args: Variable positional arguments (tuple)
   - **kwargs: Variable keyword arguments (dict)
   - Order: regular, *args, **kwargs
   - Use * to unpack lists, ** to unpack dicts

4. LAMBDA FUNCTIONS:
   - Anonymous functions: lambda x: expression
   - Use with map(), filter(), reduce(), sorted()
   - Best for simple one-line operations
   - Not for complex logic (use regular functions)

5. SCOPE (LEGB):
   - Local: Inside function
   - Enclosing: In outer function
   - Global: Module level
   - Built-in: Python's built-in functions
   - Use global to modify global variables
   - Use nonlocal to modify enclosing variables

6. TYPE HINTS:
   - Syntax: def func(param: type) -> return_type:
   - Don't enforce types (documentation only)
   - Use typing module for complex types
   - Helpful for IDE support and static analysis

7. RECURSION:
   - Must have base case (stopping condition)
   - Recursive case calls function with smaller input
   - Can cause stack overflow for deep recursion
   - Often more elegant but less efficient than iteration

8. BEST PRACTICES:
   - Write clear, descriptive function names
   - Use docstrings to document functions
   - Keep functions small and focused (single responsibility)
   - Avoid mutable default arguments
   - Use type hints for better code documentation
   - Prefer iteration over recursion for performance-critical code
   - Use lambda for simple operations, regular functions for complex logic
""")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
