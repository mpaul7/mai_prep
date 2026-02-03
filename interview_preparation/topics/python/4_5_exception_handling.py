"""
Python Exception Handling - Interview Preparation
Topic 4.5: Exception Handling

This module covers:
- try-except: Basic exception handling
- Exception Types: ValueError, TypeError, KeyError, IndexError, etc.
- Multiple Exceptions: Handling multiple exception types
- else & finally: else clause, finally clause
- Custom Exceptions: Creating custom exception classes
- raise: Raising exceptions
"""

# ============================================================================
# 1. UNDERSTANDING EXCEPTIONS
# ============================================================================

print("=" * 70)
print("1. UNDERSTANDING EXCEPTIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 What are Exceptions?
# ----------------------------------------------------------------------------
print("\n--- What are Exceptions? ---")

print("""
Exceptions are errors that occur during program execution.
They disrupt the normal flow of the program.

Exception handling allows you to:
- Catch and handle errors gracefully
- Prevent program crashes
- Provide meaningful error messages
- Clean up resources
- Continue program execution
""")


# ----------------------------------------------------------------------------
# 1.2 Exception Hierarchy
# ----------------------------------------------------------------------------
print("\n--- Exception Hierarchy ---")

print("""
BaseException
├── SystemExit
├── KeyboardInterrupt
└── Exception
    ├── StopIteration
    ├── StopAsyncIteration
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   └── OverflowError
    ├── AssertionError
    ├── AttributeError
    ├── BufferError
    ├── EOFError
    ├── ImportError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── MemoryError
    ├── NameError
    ├── OSError
    ├── ReferenceError
    ├── RuntimeError
    ├── SyntaxError
    ├── SystemError
    ├── TypeError
    └── ValueError
""")


# ============================================================================
# 2. BASIC EXCEPTION HANDLING (try-except)
# ============================================================================

print("\n" + "=" * 70)
print("2. BASIC EXCEPTION HANDLING (try-except)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic try-except
# ----------------------------------------------------------------------------
print("\n--- Basic try-except ---")

# Without exception handling (crashes)
# result = 10 / 0  # ZeroDivisionError

# With exception handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Example: Safe division
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

print(f"safe_divide(10, 2): {safe_divide(10, 2)}")  # 5.0
print(f"safe_divide(10, 0): {safe_divide(10, 0)}")  # None


# ----------------------------------------------------------------------------
# 2.2 Catching Exception with Variable
# ----------------------------------------------------------------------------
print("\n--- Catching Exception with Variable ---")

try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error occurred: {e}")
    print(f"Error type: {type(e)}")

# Accessing exception information
try:
    x = int("not a number")
except ValueError as e:
    print(f"ValueError: {e}")
    print(f"Exception args: {e.args}")


# ----------------------------------------------------------------------------
# 2.3 Catching All Exceptions (Not Recommended)
# ----------------------------------------------------------------------------
print("\n--- Catching All Exceptions (Not Recommended) ---")

try:
    result = 10 / 0
except Exception as e:
    print(f"Caught exception: {type(e).__name__}: {e}")

# Better: Catch specific exceptions
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Division error: {e}")
except Exception as e:
    print(f"Other error: {e}")


# ============================================================================
# 3. COMMON EXCEPTION TYPES
# ============================================================================

print("\n" + "=" * 70)
print("3. COMMON EXCEPTION TYPES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 ValueError
# ----------------------------------------------------------------------------
print("\n--- ValueError ---")

# Raised when function receives argument of correct type but inappropriate value
try:
    age = int("twenty")
except ValueError as e:
    print(f"ValueError: {e}")

try:
    import math
    result = math.sqrt(-1)
except ValueError as e:
    print(f"ValueError: {e}")


# ----------------------------------------------------------------------------
# 3.2 TypeError
# ----------------------------------------------------------------------------
print("\n--- TypeError ---")

# Raised when operation is performed on inappropriate type
try:
    result = "hello" + 5
except TypeError as e:
    print(f"TypeError: {e}")

try:
    result = len(42)
except TypeError as e:
    print(f"TypeError: {e}")


# ----------------------------------------------------------------------------
# 3.3 KeyError
# ----------------------------------------------------------------------------
print("\n--- KeyError ---")

# Raised when dictionary key is not found
my_dict = {"name": "Alice", "age": 25}
try:
    value = my_dict["email"]
except KeyError as e:
    print(f"KeyError: Key {e} not found")

# Safe access
value = my_dict.get("email", "N/A")
print(f"Safe access: {value}")


# ----------------------------------------------------------------------------
# 3.4 IndexError
# ----------------------------------------------------------------------------
print("\n--- IndexError ---")

# Raised when sequence index is out of range
my_list = [1, 2, 3]
try:
    value = my_list[10]
except IndexError as e:
    print(f"IndexError: {e}")

# Safe access
if len(my_list) > 10:
    value = my_list[10]
else:
    value = None


# ----------------------------------------------------------------------------
# 3.5 AttributeError
# ----------------------------------------------------------------------------
print("\n--- AttributeError ---")

# Raised when attribute doesn't exist
class Person:
    def __init__(self, name):
        self.name = name

person = Person("Alice")
try:
    age = person.age
except AttributeError as e:
    print(f"AttributeError: {e}")

# Safe access
age = getattr(person, "age", None)
print(f"Safe access: {age}")


# ----------------------------------------------------------------------------
# 3.6 NameError
# ----------------------------------------------------------------------------
print("\n--- NameError ---")

# Raised when variable name is not found
try:
    print(undefined_variable)
except NameError as e:
    print(f"NameError: {e}")


# ----------------------------------------------------------------------------
# 3.7 ZeroDivisionError
# ----------------------------------------------------------------------------
print("\n--- ZeroDivisionError ---")

try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")


# ----------------------------------------------------------------------------
# 3.8 FileNotFoundError
# ----------------------------------------------------------------------------
print("\n--- FileNotFoundError ---")

try:
    with open("nonexistent.txt", "r") as f:
        content = f.read()
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")


# ----------------------------------------------------------------------------
# 3.9 ImportError
# ----------------------------------------------------------------------------
print("\n--- ImportError ---")

try:
    import nonexistent_module
except ImportError as e:
    print(f"ImportError: {e}")


# ============================================================================
# 4. MULTIPLE EXCEPTIONS
# ============================================================================

print("\n" + "=" * 70)
print("4. MULTIPLE EXCEPTIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Multiple except Clauses
# ----------------------------------------------------------------------------
print("\n--- Multiple except Clauses ---")

def process_number(value):
    try:
        num = int(value)
        result = 10 / num
        return result
    except ValueError:
        print("Invalid number format")
    except ZeroDivisionError:
        print("Cannot divide by zero")
    except Exception as e:
        print(f"Unexpected error: {e}")

process_number("10")    # Works
process_number("abc")   # ValueError
process_number("0")     # ZeroDivisionError


# ----------------------------------------------------------------------------
# 4.2 Catching Multiple Exceptions in One Clause
# ----------------------------------------------------------------------------
print("\n--- Catching Multiple Exceptions in One Clause ---")

try:
    # Some code that might raise multiple exceptions
    value = int("not a number")
    result = 10 / value
except (ValueError, TypeError) as e:
    print(f"Value or type error: {e}")
except ZeroDivisionError as e:
    print(f"Division error: {e}")


# ----------------------------------------------------------------------------
# 4.3 Exception Order Matters
# ----------------------------------------------------------------------------
print("\n--- Exception Order Matters ---")

# More specific exceptions should come first
try:
    result = 10 / 0
except Exception as e:  # Too broad - catches everything
    print(f"Caught: {e}")
except ZeroDivisionError as e:  # Never reached!
    print(f"Division error: {e}")

# Correct order: specific first
try:
    result = 10 / 0
except ZeroDivisionError as e:  # Specific first
    print(f"Division error: {e}")
except Exception as e:  # General catch-all
    print(f"Other error: {e}")


# ============================================================================
# 5. else CLAUSE
# ============================================================================

print("\n" + "=" * 70)
print("5. else CLAUSE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 try-except-else
# ----------------------------------------------------------------------------
print("\n--- try-except-else ---")

# else clause executes if no exception occurred
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Division by zero!")
else:
    print(f"No exception occurred. Result: {result}")

# Example: File reading
try:
    # Simulate file reading
    data = "file content"
except FileNotFoundError:
    print("File not found")
else:
    print(f"File read successfully: {data}")


# ----------------------------------------------------------------------------
# 5.2 When to Use else
# ----------------------------------------------------------------------------
print("\n--- When to Use else ---")

def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    else:
        # Only executed if no exception
        print("Division successful")
        return result

print(f"divide_numbers(10, 2): {divide_numbers(10, 2)}")
print(f"divide_numbers(10, 0): {divide_numbers(10, 0)}")


# ============================================================================
# 6. finally CLAUSE
# ============================================================================

print("\n" + "=" * 70)
print("6. finally CLAUSE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 6.1 try-except-finally
# ----------------------------------------------------------------------------
print("\n--- try-except-finally ---")

# finally always executes, regardless of exceptions
try:
    result = 10 / 2
    print(f"Result: {result}")
except ZeroDivisionError:
    print("Division by zero!")
finally:
    print("This always executes")

# Even with exception
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Division by zero!")
finally:
    print("This still executes")


# ----------------------------------------------------------------------------
# 6.2 finally for Cleanup
# ----------------------------------------------------------------------------
print("\n--- finally for Cleanup ---")

# Common use: Resource cleanup
file = None
try:
    file = open("example.txt", "w")
    file.write("Hello")
    # Simulate error
    # raise ValueError("Something went wrong")
except Exception as e:
    print(f"Error: {e}")
finally:
    if file:
        file.close()
        print("File closed")


# ----------------------------------------------------------------------------
# 6.3 Complete try-except-else-finally
# ----------------------------------------------------------------------------
print("\n--- Complete try-except-else-finally ---")

def process_data(data):
    try:
        result = int(data)
    except ValueError:
        print("Invalid data format")
        return None
    else:
        print("Data processed successfully")
        return result
    finally:
        print("Cleanup: This always runs")

process_data("42")
process_data("abc")


# ============================================================================
# 7. RAISING EXCEPTIONS (raise)
# ============================================================================

print("\n" + "=" * 70)
print("7. RAISING EXCEPTIONS (raise)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 7.1 Basic raise
# ----------------------------------------------------------------------------
print("\n--- Basic raise ---")

# Raise exception
def check_positive(number):
    if number < 0:
        raise ValueError("Number must be positive")
    return number

try:
    result = check_positive(-5)
except ValueError as e:
    print(f"Error: {e}")


# ----------------------------------------------------------------------------
# 7.2 raise with Exception Instance
# ----------------------------------------------------------------------------
print("\n--- raise with Exception Instance ---")

def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age cannot exceed 150")
    return age

try:
    validate_age(-5)
except ValueError as e:
    print(f"Validation error: {e}")


# ----------------------------------------------------------------------------
# 7.3 Re-raising Exceptions
# ----------------------------------------------------------------------------
print("\n--- Re-raising Exceptions ---")

def process_value(value):
    try:
        result = int(value)
    except ValueError:
        print("Converting failed, re-raising...")
        raise  # Re-raise the same exception

try:
    process_value("abc")
except ValueError as e:
    print(f"Caught re-raised exception: {e}")


# ----------------------------------------------------------------------------
# 7.4 raise from (Exception Chaining)
# ----------------------------------------------------------------------------
print("\n--- raise from (Exception Chaining) ---")

def process_file(filename):
    try:
        with open(filename, "r") as f:
            content = f.read()
    except FileNotFoundError as e:
        raise ValueError(f"Invalid file: {filename}") from e

try:
    process_file("nonexistent.txt")
except ValueError as e:
    print(f"ValueError: {e}")
    print(f"Caused by: {e.__cause__}")


# ============================================================================
# 8. CUSTOM EXCEPTIONS
# ============================================================================

print("\n" + "=" * 70)
print("8. CUSTOM EXCEPTIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 8.1 Basic Custom Exception
# ----------------------------------------------------------------------------
print("\n--- Basic Custom Exception ---")

class CustomError(Exception):
    """Base class for custom exceptions."""
    pass

# Using custom exception
def check_value(value):
    if value < 0:
        raise CustomError("Value must be non-negative")
    return value

try:
    check_value(-5)
except CustomError as e:
    print(f"Custom error: {e}")


# ----------------------------------------------------------------------------
# 8.2 Custom Exception with Message
# ----------------------------------------------------------------------------
print("\n--- Custom Exception with Message ---")

class ValidationError(Exception):
    """Exception raised for validation errors."""
    
    def __init__(self, message, value=None):
        self.message = message
        self.value = value
        super().__init__(self.message)
    
    def __str__(self):
        if self.value is not None:
            return f"{self.message}: {self.value}"
        return self.message

def validate_email(email):
    if "@" not in email:
        raise ValidationError("Invalid email format", email)
    return email

try:
    validate_email("invalid-email")
except ValidationError as e:
    print(f"Validation error: {e}")


# ----------------------------------------------------------------------------
# 8.3 Exception Hierarchy
# ----------------------------------------------------------------------------
print("\n--- Exception Hierarchy ---")

class DataError(Exception):
    """Base exception for data-related errors."""
    pass

class InvalidDataError(DataError):
    """Raised when data is invalid."""
    pass

class MissingDataError(DataError):
    """Raised when data is missing."""
    pass

def process_data(data):
    if data is None:
        raise MissingDataError("Data is missing")
    if not isinstance(data, dict):
        raise InvalidDataError("Data must be a dictionary")
    return data

# Catching base exception catches all derived exceptions
try:
    process_data(None)
except DataError as e:
    print(f"Data error: {type(e).__name__}: {e}")

try:
    process_data("not a dict")
except DataError as e:
    print(f"Data error: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------
# 8.4 Custom Exception with Additional Attributes
# ----------------------------------------------------------------------------
print("\n--- Custom Exception with Additional Attributes ---")

class APIError(Exception):
    """Exception for API-related errors."""
    
    def __init__(self, message, status_code=None, response=None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)
    
    def __str__(self):
        if self.status_code:
            return f"{self.message} (Status: {self.status_code})"
        return self.message

def make_api_call():
    # Simulate API error
    raise APIError("API request failed", status_code=404, response={"error": "Not found"})

try:
    make_api_call()
except APIError as e:
    print(f"API Error: {e}")
    print(f"Status code: {e.status_code}")
    print(f"Response: {e.response}")


# ============================================================================
# 9. ADVANCED PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("9. ADVANCED PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 9.1 Exception Handling in Loops
# ----------------------------------------------------------------------------
print("\n--- Exception Handling in Loops ---")

numbers = ["10", "20", "abc", "30", "xyz", "40"]
valid_numbers = []

for num_str in numbers:
    try:
        num = int(num_str)
        valid_numbers.append(num)
    except ValueError:
        print(f"Skipping invalid number: {num_str}")

print(f"Valid numbers: {valid_numbers}")


# ----------------------------------------------------------------------------
# 9.2 Exception Handling in Functions
# ----------------------------------------------------------------------------
print("\n--- Exception Handling in Functions ---")

def safe_operation(func, *args, **kwargs):
    """Wrapper that handles exceptions safely."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        return None

def divide(a, b):
    return a / b

result = safe_operation(divide, 10, 0)
print(f"Result: {result}")


# ----------------------------------------------------------------------------
# 9.3 Assertions vs Exceptions
# ----------------------------------------------------------------------------
print("\n--- Assertions vs Exceptions ---")

# Assertions: For debugging, can be disabled
def calculate_average(numbers):
    assert len(numbers) > 0, "List cannot be empty"
    return sum(numbers) / len(numbers)

# Exceptions: For runtime errors, always active
def calculate_average_safe(numbers):
    if len(numbers) == 0:
        raise ValueError("List cannot be empty")
    return sum(numbers) / len(numbers)

try:
    result = calculate_average_safe([])
except ValueError as e:
    print(f"Error: {e}")


# ----------------------------------------------------------------------------
# 9.4 Context Managers and Exceptions
# ----------------------------------------------------------------------------
print("\n--- Context Managers and Exceptions ---")

class ErrorContext:
    """Context manager that handles exceptions."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"Exception handled: {exc_type.__name__}: {exc_value}")
            return True  # Suppress exception
        return False

with ErrorContext():
    raise ValueError("Test exception")
print("Code continues after exception")


# ============================================================================
# 10. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("10. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Safe List Access
print("\n--- Exercise 1: Safe List Access ---")

def safe_get_item(lst, index, default=None):
    """Safely get item from list."""
    try:
        return lst[index]
    except IndexError:
        return default

my_list = [1, 2, 3]
print(f"safe_get_item(my_list, 1): {safe_get_item(my_list, 1)}")
print(f"safe_get_item(my_list, 10): {safe_get_item(my_list, 10)}")


# Exercise 2: Input Validation
print("\n--- Exercise 2: Input Validation ---")

class InvalidInputError(Exception):
    """Exception for invalid input."""
    pass

def get_positive_integer(prompt):
    """Get positive integer from user."""
    try:
        value = int(input(prompt))
        if value <= 0:
            raise InvalidInputError("Number must be positive")
        return value
    except ValueError:
        raise InvalidInputError("Invalid number format")
    except InvalidInputError:
        raise
    except Exception as e:
        raise InvalidInputError(f"Unexpected error: {e}")


# Exercise 3: Retry with Exception Handling
print("\n--- Exercise 3: Retry with Exception Handling ---")

def retry_operation(func, max_attempts=3):
    """Retry operation on failure."""
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts:
                raise
            print(f"Attempt {attempt} failed: {e}. Retrying...")
    return None

def unreliable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "Success!"

# Uncomment to test:
# result = retry_operation(unreliable_function)
# print(f"Result: {result}")


# Exercise 4: Exception Logging
print("\n--- Exercise 4: Exception Logging ---")

def log_exception(func):
    """Decorator that logs exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Exception in {func.__name__}: {type(e).__name__}: {e}")
            raise
    return wrapper

@log_exception
def divide(a, b):
    return a / b

try:
    result = divide(10, 0)
except ZeroDivisionError:
    print("Exception caught and logged")


# ============================================================================
# 11. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("11. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between except and except Exception?
print("\n--- Q1: except vs except Exception ---")
print("""
except: Catches all exceptions (including system exits)
except Exception: Catches all standard exceptions (not system exits)

Best practice: Use except Exception or specific exceptions
""")

# Q2: When to use else vs finally?
print("\n--- Q2: else vs finally ---")
print("""
else: Executes only if no exception occurred
finally: Always executes, regardless of exceptions

Use else for code that should only run on success
Use finally for cleanup code that must always run
""")

# Q3: What happens if exception occurs in finally?
print("\n--- Q3: Exception in finally ---")
try:
    try:
        raise ValueError("Original exception")
    finally:
        # If exception occurs here, it replaces the original
        # raise TypeError("Exception in finally")
        print("Finally block")
except ValueError as e:
    print(f"Caught: {e}")


# Q4: How to create custom exceptions?
print("\n--- Q4: Creating Custom Exceptions ---")
print("""
1. Inherit from Exception (or more specific exception)
2. Optionally override __init__ and __str__
3. Add custom attributes if needed
4. Document the exception with docstring
""")


# Q5: What's exception chaining?
print("\n--- Q5: Exception Chaining ---")
print("""
Exception chaining preserves the original exception when raising a new one.
Use 'raise NewException from OriginalException'

This helps with debugging by showing the full exception chain.
""")


# ============================================================================
# 12. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("12. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. EXCEPTION HANDLING:
   - Use try-except to catch and handle exceptions
   - Catch specific exceptions when possible
   - Avoid bare except (use except Exception)
   - Use else for code that runs on success
   - Use finally for cleanup code

2. COMMON EXCEPTION TYPES:
   - ValueError: Wrong value type
   - TypeError: Wrong type
   - KeyError: Dictionary key not found
   - IndexError: List index out of range
   - AttributeError: Attribute doesn't exist
   - ZeroDivisionError: Division by zero
   - FileNotFoundError: File doesn't exist

3. MULTIPLE EXCEPTIONS:
   - Use multiple except clauses for different exceptions
   - Order matters: specific before general
   - Can catch multiple in one clause: except (A, B):

4. RAISING EXCEPTIONS:
   - Use raise to raise exceptions
   - raise Exception("message")
   - raise (re-raise current exception)
   - raise NewException from OldException (chaining)

5. CUSTOM EXCEPTIONS:
   - Inherit from Exception
   - Add custom attributes
   - Override __str__ for custom messages
   - Create exception hierarchy for organization

6. BEST PRACTICES:
   - Catch specific exceptions
   - Don't suppress exceptions silently
   - Provide meaningful error messages
   - Use finally for cleanup
   - Document custom exceptions
   - Don't use exceptions for control flow

7. COMMON PATTERNS:
   - Try-except-else-finally
   - Exception handling in loops
   - Retry logic with exceptions
   - Validation with custom exceptions
   - Resource cleanup in finally

8. MISTAKES TO AVOID:
   - Catching too broad exceptions
   - Suppressing exceptions without logging
   - Using exceptions for normal flow
   - Not cleaning up resources
   - Ignoring exception order
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
