"""
Python Debugging - Interview Preparation
Topic 8.4: Debugging

This module covers:
- print() Statements: Debugging output
- Common Errors: SyntaxError, TypeError, ValueError, KeyError, IndexError
- Error Messages: Understanding tracebacks
"""

import traceback
import sys

# ============================================================================
# 1. DEBUGGING WITH print() STATEMENTS
# ============================================================================

print("=" * 70)
print("1. DEBUGGING WITH print() STATEMENTS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Basic print() Debugging
# ----------------------------------------------------------------------------
print("\n--- 1.1 Basic print() Debugging ---")

def calculate_average(numbers):
    """Example function with print debugging"""
    print(f"DEBUG: Input numbers: {numbers}")
    print(f"DEBUG: Length: {len(numbers)}")
    
    total = sum(numbers)
    print(f"DEBUG: Total: {total}")
    
    average = total / len(numbers)
    print(f"DEBUG: Average: {average}")
    
    return average

# Test with print debugging
result = calculate_average([10, 20, 30])
print(f"Result: {result}\n")


# ----------------------------------------------------------------------------
# 1.2 Conditional print() Debugging
# ----------------------------------------------------------------------------
print("--- 1.2 Conditional print() Debugging ---")

DEBUG = True  # Toggle debugging on/off

def calculate_average_with_flag(numbers):
    """Using flag to control debug output"""
    if DEBUG:
        print(f"DEBUG: Input: {numbers}")
    
    total = sum(numbers)
    average = total / len(numbers)
    
    if DEBUG:
        print(f"DEBUG: Result: {average}")
    
    return average

result = calculate_average_with_flag([10, 20, 30])
print(f"Result: {result}\n")


# ----------------------------------------------------------------------------
# 1.3 Using f-strings for Better Debug Output
# ----------------------------------------------------------------------------
print("--- 1.3 Using f-strings for Better Debug Output ---")

def process_data(data):
    """Better debug output with f-strings"""
    print(f"[DEBUG] Function: process_data")
    print(f"[DEBUG] Input type: {type(data)}")
    print(f"[DEBUG] Input value: {data}")
    print(f"[DEBUG] Input length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    
    # Process data
    result = [x * 2 for x in data]
    
    print(f"[DEBUG] Output: {result}")
    return result

process_data([1, 2, 3])
print()


# ----------------------------------------------------------------------------
# 1.4 Debugging Loops
# ----------------------------------------------------------------------------
print("--- 1.4 Debugging Loops ---")

def find_max(numbers):
    """Debugging a loop"""
    if not numbers:
        return None
    
    max_val = numbers[0]
    print(f"DEBUG: Initial max_val: {max_val}")
    
    for i, num in enumerate(numbers):
        print(f"DEBUG: Iteration {i}, num={num}, current max={max_val}")
        if num > max_val:
            print(f"DEBUG: Found new max: {num}")
            max_val = num
    
    print(f"DEBUG: Final max_val: {max_val}")
    return max_val

find_max([3, 1, 4, 1, 5, 9, 2])
print()


# ----------------------------------------------------------------------------
# 1.5 Debugging with Variable Names
# ----------------------------------------------------------------------------
print("--- 1.5 Debugging with Variable Names ---")

def debug_variables():
    """Show variable values clearly"""
    x = 10
    y = 20
    z = x + y
    
    # Good: Shows variable name and value
    print(f"DEBUG: x = {x}")
    print(f"DEBUG: y = {y}")
    print(f"DEBUG: z = {z}")
    
    # Even better: Use repr() for exact representation
    name = "Alice"
    print(f"DEBUG: name = {repr(name)}")
    print(f"DEBUG: name type = {type(name)}")

debug_variables()
print()


# ----------------------------------------------------------------------------
# 1.6 Debugging Function Calls
# ----------------------------------------------------------------------------
print("--- 1.6 Debugging Function Calls ---")

def add(a, b):
    """Simple function to debug"""
    print(f"DEBUG: add() called with a={a}, b={b}")
    result = a + b
    print(f"DEBUG: add() returning {result}")
    return result

def multiply(x, y):
    """Function that calls another function"""
    print(f"DEBUG: multiply() called with x={x}, y={y}")
    sum_result = add(x, y)
    print(f"DEBUG: multiply() got sum={sum_result}")
    result = sum_result * 2
    print(f"DEBUG: multiply() returning {result}")
    return result

multiply(5, 3)
print()


# ============================================================================
# 2. COMMON ERRORS
# ============================================================================

print("=" * 70)
print("2. COMMON ERRORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 SyntaxError
# ----------------------------------------------------------------------------
print("\n--- 2.1 SyntaxError ---")
print("""
SyntaxError occurs when Python cannot parse your code.
Common causes:
- Missing colons (:)
- Unmatched parentheses, brackets, or quotes
- Incorrect indentation
- Invalid syntax
""")

# Example 1: Missing colon
print("Example 1: Missing colon")
try:
    exec("""
if True
    print('Missing colon')
""")
except SyntaxError as e:
    print(f"  SyntaxError: {e}")
    print(f"  Line: {e.lineno}")
    print(f"  Text: {e.text}")

# Example 2: Unmatched parentheses
print("\nExample 2: Unmatched parentheses")
try:
    exec("result = (1 + 2")
except SyntaxError as e:
    print(f"  SyntaxError: {e}")

# Example 3: Invalid indentation
print("\nExample 3: Invalid indentation")
try:
    exec("""
if True:
print('Wrong indentation')
""")
except IndentationError as e:
    print(f"  IndentationError: {e}")
    print(f"  Line: {e.lineno}")


# ----------------------------------------------------------------------------
# 2.2 TypeError
# ----------------------------------------------------------------------------
print("\n--- 2.2 TypeError ---")
print("""
TypeError occurs when operation is performed on inappropriate type.
Common causes:
- Wrong type passed to function
- Operation not supported for type
- Missing required arguments
""")

# Example 1: Wrong type in operation
print("Example 1: Wrong type in operation")
try:
    result = "hello" + 5
except TypeError as e:
    print(f"  TypeError: {e}")

# Example 2: Wrong type passed to function
print("\nExample 2: Wrong type passed to function")
try:
    result = len(42)
except TypeError as e:
    print(f"  TypeError: {e}")

# Example 3: Missing required arguments
print("\nExample 3: Missing required arguments")
try:
    def add(a, b):
        return a + b
    result = add(5)  # Missing b
except TypeError as e:
    print(f"  TypeError: {e}")

# Example 4: Wrong number of arguments
print("\nExample 4: Wrong number of arguments")
try:
    result = int("10", 2, 3)  # Too many arguments
except TypeError as e:
    print(f"  TypeError: {e}")


# ----------------------------------------------------------------------------
# 2.3 ValueError
# ----------------------------------------------------------------------------
print("\n--- 2.3 ValueError ---")
print("""
ValueError occurs when function receives correct type but inappropriate value.
Common causes:
- Invalid value for conversion (e.g., int("abc"))
- Value out of range
- Invalid format
""")

# Example 1: Invalid conversion
print("Example 1: Invalid conversion")
try:
    age = int("twenty")
except ValueError as e:
    print(f"  ValueError: {e}")

# Example 2: Invalid value for function
print("\nExample 2: Invalid value for function")
try:
    import math
    result = math.sqrt(-1)
except ValueError as e:
    print(f"  ValueError: {e}")

# Example 3: Invalid list index value
print("\nExample 3: Invalid list index value")
try:
    my_list = [1, 2, 3]
    index = int("not a number")
    value = my_list[index]
except ValueError as e:
    print(f"  ValueError: {e}")


# ----------------------------------------------------------------------------
# 2.4 KeyError
# ----------------------------------------------------------------------------
print("\n--- 2.4 KeyError ---")
print("""
KeyError occurs when dictionary key is not found.
Common causes:
- Accessing non-existent key
- Typo in key name
- Key not set
""")

# Example 1: Accessing non-existent key
print("Example 1: Accessing non-existent key")
try:
    my_dict = {"name": "Alice", "age": 25}
    email = my_dict["email"]
except KeyError as e:
    print(f"  KeyError: {e}")
    print(f"  Key '{e}' not found in dictionary")

# Example 2: Typo in key
print("\nExample 2: Typo in key")
try:
    my_dict = {"first_name": "John", "last_name": "Doe"}
    name = my_dict["firstname"]  # Typo: firstname vs first_name
except KeyError as e:
    print(f"  KeyError: {e}")

# Example 3: Safe access methods
print("\nExample 3: Safe access methods")
my_dict = {"name": "Alice", "age": 25}

# Method 1: get() with default
email = my_dict.get("email", "N/A")
print(f"  Using get(): {email}")

# Method 2: Check before access
if "email" in my_dict:
    email = my_dict["email"]
else:
    email = "N/A"
print(f"  Using 'in' check: {email}")

# Method 3: setdefault()
my_dict.setdefault("email", "default@example.com")
print(f"  Using setdefault(): {my_dict['email']}")


# ----------------------------------------------------------------------------
# 2.5 IndexError
# ----------------------------------------------------------------------------
print("\n--- 2.5 IndexError ---")
print("""
IndexError occurs when sequence index is out of range.
Common causes:
- Accessing index beyond list length
- Negative index too large
- Empty list access
""")

# Example 1: Index out of range
print("Example 1: Index out of range")
try:
    my_list = [1, 2, 3]
    value = my_list[10]
except IndexError as e:
    print(f"  IndexError: {e}")

# Example 2: Negative index too large
print("\nExample 2: Negative index too large")
try:
    my_list = [1, 2, 3]
    value = my_list[-10]
except IndexError as e:
    print(f"  IndexError: {e}")

# Example 3: Empty list access
print("\nExample 3: Empty list access")
try:
    my_list = []
    value = my_list[0]
except IndexError as e:
    print(f"  IndexError: {e}")

# Example 4: Safe access methods
print("\nExample 4: Safe access methods")
my_list = [1, 2, 3]

# Method 1: Check length
if len(my_list) > 10:
    value = my_list[10]
else:
    value = None
print(f"  Using length check: {value}")

# Method 2: Try-except
try:
    value = my_list[10]
except IndexError:
    value = None
print(f"  Using try-except: {value}")

# Method 3: Slice (returns empty list if out of range)
value = my_list[10:11]  # Returns [] if index out of range
print(f"  Using slice: {value}")


# ----------------------------------------------------------------------------
# 2.6 Other Common Errors
# ----------------------------------------------------------------------------
print("\n--- 2.6 Other Common Errors ---")

# AttributeError
print("AttributeError: Attribute doesn't exist")
try:
    my_list = [1, 2, 3]
    result = my_list.nonexistent_method()
except AttributeError as e:
    print(f"  AttributeError: {e}")

# NameError
print("\nNameError: Variable not defined")
try:
    result = undefined_variable
except NameError as e:
    print(f"  NameError: {e}")

# ZeroDivisionError
print("\nZeroDivisionError: Division by zero")
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"  ZeroDivisionError: {e}")

# FileNotFoundError
print("\nFileNotFoundError: File doesn't exist")
try:
    with open("nonexistent_file.txt") as f:
        content = f.read()
except FileNotFoundError as e:
    print(f"  FileNotFoundError: {e}")


# ============================================================================
# 3. UNDERSTANDING TRACEBACKS
# ============================================================================

print("\n" + "=" * 70)
print("3. UNDERSTANDING TRACEBACKS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 What is a Traceback?
# ----------------------------------------------------------------------------
print("\n--- 3.1 What is a Traceback? ---")
print("""
A traceback shows the call stack when an error occurs.
It tells you:
- Where the error occurred (file, line number)
- What function called what function
- The sequence of function calls leading to the error
- The error message and type
""")


# ----------------------------------------------------------------------------
# 3.2 Reading a Traceback
# ----------------------------------------------------------------------------
print("\n--- 3.2 Reading a Traceback ---")

def function_a():
    """Function that calls function_b"""
    print("  In function_a()")
    return function_b()

def function_b():
    """Function that calls function_c"""
    print("  In function_b()")
    return function_c()

def function_c():
    """Function that raises an error"""
    print("  In function_c()")
    raise ValueError("Something went wrong!")

print("Example traceback:")
try:
    function_a()
except ValueError as e:
    print("\nTraceback (most recent call last):")
    print("  File 'example.py', line 1, in <module>")
    print("    function_a()")
    print("  File 'example.py', line 2, in function_a")
    print("    return function_b()")
    print("  File 'example.py', line 3, in function_b")
    print("    return function_c()")
    print("  File 'example.py', line 4, in function_c")
    print("    raise ValueError('Something went wrong!')")
    print(f"ValueError: {e}")
    print("\nReading from bottom to top:")
    print("  1. Error type and message at bottom")
    print("  2. Most recent call at bottom of stack")
    print("  3. Original call at top of stack")


# ----------------------------------------------------------------------------
# 3.3 Getting Full Traceback Information
# ----------------------------------------------------------------------------
print("\n--- 3.3 Getting Full Traceback Information ---")

def demonstrate_traceback():
    """Function that causes an error"""
    x = 10
    y = 0
    result = x / y  # ZeroDivisionError
    return result

try:
    demonstrate_traceback()
except ZeroDivisionError:
    print("Full traceback information:")
    print(traceback.format_exc())


# ----------------------------------------------------------------------------
# 3.4 Common Traceback Patterns
# ----------------------------------------------------------------------------
print("\n--- 3.4 Common Traceback Patterns ---")

# Pattern 1: Direct error in your code
print("Pattern 1: Direct error in your code")
try:
    my_list = [1, 2, 3]
    value = my_list[10]
except IndexError as e:
    print(f"  Error: {type(e).__name__}: {e}")
    print("  Fix: Check list length before accessing index")

# Pattern 2: Error in called function
print("\nPattern 2: Error in called function")
def helper_function(data):
    return data["key"]

try:
    result = helper_function({})  # Empty dict
except KeyError as e:
    print(f"  Error: {type(e).__name__}: {e}")
    print("  Fix: Check if key exists or use .get()")

# Pattern 3: Error in imported module
print("\nPattern 3: Error in imported module")
try:
    import math
    result = math.sqrt(-1)
except ValueError as e:
    print(f"  Error: {type(e).__name__}: {e}")
    print("  Fix: Validate input before calling function")


# ----------------------------------------------------------------------------
# 3.5 Extracting Information from Traceback
# ----------------------------------------------------------------------------
print("\n--- 3.5 Extracting Information from Traceback ---")

def extract_traceback_info():
    """Extract information from traceback"""
    try:
        x = int("not a number")
    except ValueError:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        print("Exception information:")
        print(f"  Type: {exc_type.__name__}")
        print(f"  Message: {exc_value}")
        print(f"  Traceback object: {exc_traceback}")
        
        # Get traceback frames
        tb = exc_traceback
        while tb:
            print(f"\n  Frame: {tb.tb_frame.f_code.co_filename}")
            print(f"    Line: {tb.tb_lineno}")
            print(f"    Function: {tb.tb_frame.f_code.co_name}")
            tb = tb.tb_next

extract_traceback_info()


# ============================================================================
# 4. DEBUGGING STRATEGIES
# ============================================================================

print("\n" + "=" * 70)
print("4. DEBUGGING STRATEGIES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Systematic Debugging Approach
# ----------------------------------------------------------------------------
print("\n--- 4.1 Systematic Debugging Approach ---")
print("""
1. READ THE ERROR MESSAGE:
   - Understand what error occurred
   - Read the full traceback
   - Identify the line number

2. REPRODUCE THE ERROR:
   - Run the code again
   - Check if error is consistent
   - Identify the minimal case

3. ADD DEBUG STATEMENTS:
   - Print variable values before error
   - Print function arguments
   - Print intermediate results

4. ISOLATE THE PROBLEM:
   - Comment out suspicious code
   - Test parts separately
   - Use minimal test case

5. FIX AND TEST:
   - Make one change at a time
   - Test after each change
   - Verify fix works
""")


# ----------------------------------------------------------------------------
# 4.2 Common Debugging Patterns
# ----------------------------------------------------------------------------
print("\n--- 4.2 Common Debugging Patterns ---")

# Pattern 1: Check input values
def safe_divide(a, b):
    """Pattern: Validate inputs"""
    print(f"DEBUG: a={a}, b={b}, type(a)={type(a)}, type(b)={type(b)}")
    
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    
    if b == 0:
        raise ValueError("Cannot divide by zero")
    
    return a / b

# Pattern 2: Check intermediate values
def process_list(numbers):
    """Pattern: Check intermediate steps"""
    print(f"DEBUG: Input: {numbers}")
    
    filtered = [x for x in numbers if x > 0]
    print(f"DEBUG: Filtered: {filtered}")
    
    doubled = [x * 2 for x in filtered]
    print(f"DEBUG: Doubled: {doubled}")
    
    return doubled

# Pattern 3: Check function return values
def calculate_total(items):
    """Pattern: Verify return value"""
    total = sum(items)
    print(f"DEBUG: Items: {items}")
    print(f"DEBUG: Total: {total}")
    print(f"DEBUG: Expected range: 0-100")
    
    if total < 0 or total > 100:
        print(f"DEBUG: WARNING - Total out of expected range!")
    
    return total


# ----------------------------------------------------------------------------
# 4.3 Debugging Tips
# ----------------------------------------------------------------------------
print("\n--- 4.3 Debugging Tips ---")
print("""
1. USE DESCRIPTIVE DEBUG MESSAGES:
   - Include variable names
   - Include values
   - Include context

2. DEBUG AT THE RIGHT LEVEL:
   - Start with high-level overview
   - Drill down to specific areas
   - Don't debug everything at once

3. USE CONDITIONAL DEBUGGING:
   - Use flags to enable/disable
   - Use different debug levels
   - Remove debug code before production

4. READ ERRORS CAREFULLY:
   - Error messages are helpful
   - Traceback shows call stack
   - Line numbers point to problem

5. TEST INCREMENTALLY:
   - Test small pieces
   - Verify each step
   - Build up complexity gradually

6. USE PRINT EFFECTIVELY:
   - Print before and after operations
   - Print in loops to see iterations
   - Print function entry/exit
""")


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Identify the error
print("\n--- Exercise 1: Identify the Error ---")
print("""
Code:
    def calculate_average(numbers):
        total = sum(numbers)
        return total / len(numbers)
    
    result = calculate_average([])

Error: ZeroDivisionError
Reason: len([]) is 0, division by zero
Fix: Check if list is empty before dividing
""")

# Exercise 2: Fix KeyError
print("\n--- Exercise 2: Fix KeyError ---")
def get_user_email(user_dict):
    """Fix this function to handle missing key"""
    # Original (causes KeyError):
    # return user_dict["email"]
    
    # Fixed version:
    return user_dict.get("email", "N/A")

user = {"name": "Alice", "age": 25}
print(f"User email: {get_user_email(user)}")

# Exercise 3: Fix IndexError
print("\n--- Exercise 3: Fix IndexError ---")
def get_first_item(items):
    """Fix this function to handle empty list"""
    # Original (causes IndexError):
    # return items[0]
    
    # Fixed version:
    if len(items) > 0:
        return items[0]
    return None

print(f"First item: {get_first_item([1, 2, 3])}")
print(f"First item (empty): {get_first_item([])}")

# Exercise 4: Debug a function
print("\n--- Exercise 4: Debug a Function ---")
def find_max_value(data):
    """Debug this function"""
    print(f"DEBUG: Input data: {data}")
    print(f"DEBUG: Type: {type(data)}")
    
    if not data:
        print("DEBUG: Empty data, returning None")
        return None
    
    max_val = data[0]
    print(f"DEBUG: Initial max_val: {max_val}")
    
    for i, value in enumerate(data):
        print(f"DEBUG: Iteration {i}, value={value}, current max={max_val}")
        if value > max_val:
            print(f"DEBUG: New max found: {value}")
            max_val = value
    
    print(f"DEBUG: Final max_val: {max_val}")
    return max_val

find_max_value([3, 1, 4, 1, 5])


# ============================================================================
# 6. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("6. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. PRINT() DEBUGGING:
   - Use print() to see variable values
   - Print before and after operations
   - Use descriptive debug messages
   - Use flags to enable/disable debugging
   - Remove debug code before production

2. COMMON ERRORS:
   - SyntaxError: Invalid syntax (missing colon, unmatched brackets)
   - TypeError: Wrong type (string + int, len(int))
   - ValueError: Wrong value (int("abc"), sqrt(-1))
   - KeyError: Dictionary key not found
   - IndexError: List index out of range
   - AttributeError: Attribute doesn't exist
   - NameError: Variable not defined
   - ZeroDivisionError: Division by zero

3. UNDERSTANDING TRACEBACKS:
   - Read from bottom to top
   - Bottom shows error type and message
   - Stack shows function call sequence
   - Line numbers point to problem location
   - Most recent call is at bottom

4. DEBUGGING STRATEGY:
   - Read error message carefully
   - Reproduce the error
   - Add debug statements
   - Isolate the problem
   - Fix and test incrementally

5. SAFE CODING PRACTICES:
   - Check inputs before using
   - Use .get() for dictionaries
   - Check length before indexing
   - Validate types and values
   - Handle edge cases

6. INTERVIEW TIPS:
   - Explain your debugging process
   - Show how you'd add debug statements
   - Read tracebacks carefully
   - Identify error types quickly
   - Suggest fixes for common errors
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Debugging Guide Ready!")
    print("=" * 70)
