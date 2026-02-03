"""
Python Decorators - Interview Preparation
Topic 4.2: Decorators

This module covers:
- Function Decorators: @decorator syntax
- Decorator Functions: Writing custom decorators
- Decorator Classes: Class-based decorators
- Built-in Decorators: @property, @staticmethod, @classmethod, @functools.lru_cache
"""

import time
import functools
from functools import wraps, lru_cache

# ============================================================================
# 1. UNDERSTANDING DECORATORS
# ============================================================================

print("=" * 70)
print("1. UNDERSTANDING DECORATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 What are Decorators?
# ----------------------------------------------------------------------------
print("\n--- What are Decorators? ---")

print("""
Decorators are functions that modify or enhance other functions.
They allow you to wrap a function with additional functionality
without modifying the function's code directly.

Key concepts:
- Functions are first-class objects (can be passed as arguments)
- Decorators use closures to maintain state
- @decorator syntax is syntactic sugar
""")


# ----------------------------------------------------------------------------
# 1.2 Functions as First-Class Objects
# ----------------------------------------------------------------------------
print("\n--- Functions as First-Class Objects ---")

def greet(name):
    return f"Hello, {name}!"

# Functions can be assigned to variables
say_hello = greet
print(f"say_hello('Alice'): {say_hello('Alice')}")

# Functions can be passed as arguments
def call_function(func, arg):
    return func(arg)

result = call_function(greet, "Bob")
print(f"call_function(greet, 'Bob'): {result}")

# Functions can be returned from functions
def get_greeter():
    return greet

greeter = get_greeter()
print(f"greeter('Charlie'): {greeter('Charlie')}")


# ============================================================================
# 2. FUNCTION DECORATORS (@decorator syntax)
# ============================================================================

print("\n" + "=" * 70)
print("2. FUNCTION DECORATORS (@decorator syntax)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic Decorator Syntax
# ----------------------------------------------------------------------------
print("\n--- Basic Decorator Syntax ---")

# Simple decorator function
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

# Using decorator syntax
@my_decorator
def say_hello():
    print("Hello!")

print("Calling decorated function:")
say_hello()

# Equivalent to:
# say_hello = my_decorator(say_hello)

# Without decorator syntax (manual wrapping)
def say_goodbye():
    print("Goodbye!")

say_goodbye = my_decorator(say_goodbye)
print("\nManually decorated function:")
say_goodbye()


# ----------------------------------------------------------------------------
# 2.2 Decorator with Arguments
# ----------------------------------------------------------------------------
print("\n--- Decorator with Arguments ---")

def decorator_with_args(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@decorator_with_args
def add(a, b):
    return a + b

result = add(3, 5)
print(f"Result: {result}")


# ============================================================================
# 3. WRITING CUSTOM DECORATORS
# ============================================================================

print("\n" + "=" * 70)
print("3. WRITING CUSTOM DECORATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Timer Decorator
# ----------------------------------------------------------------------------
print("\n--- Timer Decorator ---")

def timer(func):
    """Decorator that measures function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.1)
    return "Done"

result = slow_function()


# ----------------------------------------------------------------------------
# 3.2 Preserving Function Metadata with @wraps
# ----------------------------------------------------------------------------
print("\n--- Preserving Function Metadata with @wraps ---")

# Without @wraps (loses metadata)
def decorator_no_wraps(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator_no_wraps
def my_function():
    """This is my function."""
    pass

print(f"Without @wraps - name: {my_function.__name__}")  # 'wrapper'
print(f"Without @wraps - doc: {my_function.__doc__}")  # None

# With @wraps (preserves metadata)
def decorator_with_wraps(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator_with_wraps
def my_function2():
    """This is my function."""
    pass

print(f"With @wraps - name: {my_function2.__name__}")  # 'my_function2'
print(f"With @wraps - doc: {my_function2.__doc__}")  # 'This is my function.'


# ----------------------------------------------------------------------------
# 3.3 Decorator with Arguments (Decorator Factory)
# ----------------------------------------------------------------------------
print("\n--- Decorator with Arguments (Decorator Factory) ---")

def repeat(times):
    """Decorator that repeats function call multiple times."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    return f"Hello, {name}!"

results = greet("Alice")
print(f"Repeated 3 times: {results}")


# ----------------------------------------------------------------------------
# 3.4 Conditional Decorator
# ----------------------------------------------------------------------------
print("\n--- Conditional Decorator ---")

def debug(debug_mode=True):
    """Decorator that prints debug info if debug_mode is True."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if debug_mode:
                print(f"DEBUG: Calling {func.__name__} with args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            if debug_mode:
                print(f"DEBUG: {func.__name__} returned {result}")
            return result
        return wrapper
    return decorator

@debug(debug_mode=True)
def multiply(a, b):
    return a * b

result = multiply(3, 4)


# ----------------------------------------------------------------------------
# 3.5 Retry Decorator
# ----------------------------------------------------------------------------
print("\n--- Retry Decorator ---")

def retry(max_attempts=3, delay=1):
    """Decorator that retries function on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.1)
def unreliable_function():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ValueError("Random failure")
    return "Success!"

try:
    result = unreliable_function()
    print(f"Result: {result}")
except ValueError as e:
    print(f"Failed after retries: {e}")


# ----------------------------------------------------------------------------
# 3.6 Validation Decorator
# ----------------------------------------------------------------------------
print("\n--- Validation Decorator ---")

def validate_types(*expected_types):
    """Decorator that validates argument types."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check positional arguments
            for i, (arg, expected_type) in enumerate(zip(args, expected_types)):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Argument {i} must be {expected_type.__name__}, got {type(arg).__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(int, int)
def add(a, b):
    return a + b

result = add(3, 5)
print(f"add(3, 5) = {result}")

try:
    add("3", 5)  # Should raise TypeError
except TypeError as e:
    print(f"Validation error: {e}")


# ----------------------------------------------------------------------------
# 3.7 Caching Decorator (Simple)
# ----------------------------------------------------------------------------
print("\n--- Caching Decorator (Simple) ---")

def simple_cache(func):
    """Simple caching decorator."""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        if key in cache:
            print(f"Cache hit for {func.__name__}")
            return cache[key]
        print(f"Cache miss for {func.__name__}")
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    
    return wrapper

@simple_cache
def expensive_function(n):
    """Simulate expensive computation."""
    time.sleep(0.1)
    return n * 2

print(f"First call: {expensive_function(5)}")
print(f"Second call (cached): {expensive_function(5)}")


# ============================================================================
# 4. CLASS-BASED DECORATORS
# ============================================================================

print("\n" + "=" * 70)
print("4. CLASS-BASED DECORATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Basic Class Decorator
# ----------------------------------------------------------------------------
print("\n--- Basic Class Decorator ---")

class CountCalls:
    """Decorator class that counts function calls."""
    
    def __init__(self, func):
        self.func = func
        self.count = 0
        # Preserve function metadata
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello(name):
    return f"Hello, {name}!"

print(say_hello("Alice"))
print(say_hello("Bob"))
print(f"Total calls: {say_hello.count}")


# ----------------------------------------------------------------------------
# 4.2 Class Decorator with State
# ----------------------------------------------------------------------------
print("\n--- Class Decorator with State ---")

class Timer:
    """Class decorator that tracks execution time."""
    
    def __init__(self, func):
        self.func = func
        self.call_count = 0
        self.total_time = 0
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.call_count += 1
        self.total_time += execution_time
        
        print(f"{self.func.__name__} executed in {execution_time:.4f}s")
        return result
    
    def get_stats(self):
        """Get execution statistics."""
        avg_time = self.total_time / self.call_count if self.call_count > 0 else 0
        return {
            'call_count': self.call_count,
            'total_time': self.total_time,
            'average_time': avg_time
        }

@Timer
def slow_operation():
    time.sleep(0.05)
    return "Done"

slow_operation()
slow_operation()
stats = slow_operation.get_stats()
print(f"Statistics: {stats}")


# ----------------------------------------------------------------------------
# 4.3 Class Decorator with Arguments
# ----------------------------------------------------------------------------
print("\n--- Class Decorator with Arguments ---")

class Retry:
    """Class-based retry decorator."""
    
    def __init__(self, max_attempts=3, delay=1):
        self.max_attempts = max_attempts
        self.delay = delay
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == self.max_attempts:
                        raise
                    print(f"Attempt {attempt} failed. Retrying...")
                    time.sleep(self.delay)
        return wrapper

@Retry(max_attempts=3, delay=0.1)
def might_fail():
    import random
    if random.random() < 0.5:
        raise ValueError("Random failure")
    return "Success!"

try:
    result = might_fail()
    print(f"Result: {result}")
except ValueError as e:
    print(f"Failed: {e}")


# ============================================================================
# 5. BUILT-IN DECORATORS
# ============================================================================

print("\n" + "=" * 70)
print("5. BUILT-IN DECORATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 @property
# ----------------------------------------------------------------------------
print("\n--- @property Decorator ---")

class Circle:
    """Example using @property decorator."""
    
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        """Get radius."""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Set radius with validation."""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        """Calculate area (read-only property)."""
        return 3.14159 * self._radius ** 2
    
    @property
    def diameter(self):
        """Calculate diameter (read-only property)."""
        return 2 * self._radius

circle = Circle(5)
print(f"Radius: {circle.radius}")
print(f"Area: {circle.area:.2f}")
print(f"Diameter: {circle.diameter}")

# Using setter
circle.radius = 10
print(f"New radius: {circle.radius}")
print(f"New area: {circle.area:.2f}")

# Try to set negative radius
try:
    circle.radius = -5
except ValueError as e:
    print(f"Error: {e}")


# ----------------------------------------------------------------------------
# 5.2 @staticmethod
# ----------------------------------------------------------------------------
print("\n--- @staticmethod Decorator ---")

class MathUtils:
    """Example using @staticmethod."""
    
    @staticmethod
    def add(a, b):
        """Add two numbers."""
        return a + b
    
    @staticmethod
    def multiply(a, b):
        """Multiply two numbers."""
        return a * b
    
    def instance_method(self):
        """Regular instance method."""
        return "Instance method"

# Can call without creating instance
result1 = MathUtils.add(3, 5)
print(f"MathUtils.add(3, 5) = {result1}")

result2 = MathUtils.multiply(4, 6)
print(f"MathUtils.multiply(4, 6) = {result2}")

# Can also call on instance
utils = MathUtils()
result3 = utils.add(2, 3)
print(f"utils.add(2, 3) = {result3}")


# ----------------------------------------------------------------------------
# 5.3 @classmethod
# ----------------------------------------------------------------------------
print("\n--- @classmethod Decorator ---")

class Person:
    """Example using @classmethod."""
    
    population = 0
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        Person.population += 1
    
    @classmethod
    def from_birth_year(cls, name, birth_year):
        """Alternative constructor using birth year."""
        current_year = 2024
        age = current_year - birth_year
        return cls(name, age)
    
    @classmethod
    def get_population(cls):
        """Get total population."""
        return cls.population
    
    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age})"

# Regular constructor
person1 = Person("Alice", 25)
print(f"person1: {person1}")

# Using classmethod as alternative constructor
person2 = Person.from_birth_year("Bob", 1995)
print(f"person2: {person2}")

# Using classmethod to get class-level data
print(f"Total population: {Person.get_population()}")


# ----------------------------------------------------------------------------
# 5.4 @functools.lru_cache
# ----------------------------------------------------------------------------
print("\n--- @functools.lru_cache Decorator ---")

@lru_cache(maxsize=128)
def fibonacci(n):
    """Calculate Fibonacci number with caching."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# First call (computes and caches)
print("Computing Fibonacci(10)...")
result1 = fibonacci(10)
print(f"fibonacci(10) = {result1}")

# Second call (uses cache)
print("Computing Fibonacci(10) again (should use cache)...")
result2 = fibonacci(10)
print(f"fibonacci(10) = {result2}")

# Cache info
print(f"Cache info: {fibonacci.cache_info()}")

# Clear cache
fibonacci.cache_clear()
print("Cache cleared")


# ----------------------------------------------------------------------------
# 5.5 @functools.wraps (already covered, but showing usage)
# ----------------------------------------------------------------------------
print("\n--- @functools.wraps ---")

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function."""
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def example_function():
    """Example function with decorator."""
    pass

print(f"Function name: {example_function.__name__}")  # 'example_function'
print(f"Function doc: {example_function.__doc__}")  # 'Example function with decorator.'


# ============================================================================
# 6. STACKING DECORATORS
# ============================================================================

print("\n" + "=" * 70)
print("6. STACKING DECORATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 6.1 Multiple Decorators
# ----------------------------------------------------------------------------
print("\n--- Multiple Decorators ---")

def bold(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

# Stacking decorators (applied bottom to top)
@bold
@italic
def greet(name):
    return f"Hello, {name}!"

result = greet("Alice")
print(f"Decorated greeting: {result}")
# Equivalent to: greet = bold(italic(greet))


# ----------------------------------------------------------------------------
# 6.2 Decorator Order Matters
# ----------------------------------------------------------------------------
print("\n--- Decorator Order Matters ---")

@timer
@CountCalls
def example_function():
    time.sleep(0.01)
    return "Done"

result = example_function()
# Timer wraps CountCalls, so timing includes counting overhead


# ============================================================================
# 7. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Logging Decorator
print("\n--- Exercise 1: Logging Decorator ---")
def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"LOG: Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"LOG: {func.__name__} returned {result}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

result = add(3, 5)


# Exercise 2: Rate Limiting Decorator
print("\n--- Exercise 2: Rate Limiting Decorator ---")
def rate_limit(calls_per_second=1):
    """Limit function calls to specified rate."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_second=2)
def api_call():
    return "API response"

print("Making 3 API calls with rate limiting:")
for i in range(3):
    print(f"Call {i+1}: {api_call()}")


# Exercise 3: Memoization Decorator
print("\n--- Exercise 3: Memoization Decorator ---")
def memoize(func):
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache = cache
    return wrapper

@memoize
def expensive_computation(n):
    """Simulate expensive computation."""
    time.sleep(0.1)
    return n * 2

print(f"First call: {expensive_computation(10)}")
print(f"Second call (cached): {expensive_computation(10)}")
print(f"Cache size: {len(expensive_computation.cache)}")


# ============================================================================
# 8. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("8. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What is a decorator?
print("\n--- Q1: What is a decorator? ---")
print("""
A decorator is a function that takes another function as input and
extends or modifies its behavior without explicitly modifying it.
Decorators use closures to maintain state and are applied using @ syntax.
""")

# Q2: How do decorators work?
print("\n--- Q2: How do decorators work? ---")
print("""
1. Decorator function receives the function to be decorated
2. Returns a wrapper function that adds functionality
3. Wrapper function calls the original function
4. @decorator syntax is syntactic sugar for: func = decorator(func)
""")

# Q3: Why use @wraps?
print("\n--- Q3: Why use @wraps? ---")
print("""
@wraps preserves the original function's metadata (__name__, __doc__, etc.)
Without it, the wrapper function's metadata would be used instead,
which can break introspection tools and documentation generators.
""")

# Q4: What's the difference between @staticmethod and @classmethod?
print("\n--- Q4: @staticmethod vs @classmethod ---")
print("""
@staticmethod:
- Doesn't receive self or cls
- Can't access class or instance attributes
- Called on class or instance
- Use for utility functions related to the class

@classmethod:
- Receives cls (the class) as first argument
- Can access class attributes
- Can create alternative constructors
- Use for factory methods or class-level operations
""")

# Q5: When would you use a class-based decorator?
print("\n--- Q5: When to use class-based decorator? ---")
print("""
Use class-based decorators when:
- You need to maintain complex state
- You want to provide additional methods/properties
- You need decorator with arguments (cleaner syntax)
- You want to combine decorator functionality with class features
""")


# ============================================================================
# 9. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("9. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. DECORATOR BASICS:
   - Functions are first-class objects (can be passed, returned)
   - Decorators wrap functions to add functionality
   - @decorator syntax: @decorator def func(): pass
   - Equivalent to: func = decorator(func)

2. WRITING DECORATORS:
   - Decorator function takes func as argument
   - Returns wrapper function that calls func
   - Use @wraps(func) to preserve metadata
   - Handle *args, **kwargs for flexibility

3. DECORATOR WITH ARGUMENTS:
   - Create decorator factory (function that returns decorator)
   - Outer function takes decorator arguments
   - Inner function takes func and returns wrapper

4. CLASS-BASED DECORATORS:
   - Implement __init__ to receive func
   - Implement __call__ to make instance callable
   - Can maintain state between calls
   - Use functools.update_wrapper to preserve metadata

5. BUILT-IN DECORATORS:
   - @property: Create getters/setters, computed properties
   - @staticmethod: Utility methods (no self/cls)
   - @classmethod: Factory methods, class-level operations
   - @functools.lru_cache: Memoization for expensive functions
   - @functools.wraps: Preserve function metadata

6. COMMON PATTERNS:
   - Timer: Measure execution time
   - Retry: Retry on failure
   - Cache: Store results
   - Validation: Check arguments
   - Logging: Log function calls

7. BEST PRACTICES:
   - Always use @wraps to preserve metadata
   - Handle *args, **kwargs for flexibility
   - Document decorator behavior
   - Consider performance implications
   - Use class decorators for complex state management
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
