"""
Python Core Data Types - Interview Preparation
Topic 1.1: Core Data Types

This module covers:
- Primitive Types: int, float, str, bool, None
- Type Conversion: int(), float(), str(), bool()
- Type Checking: isinstance(), type()
- Immutability: Understanding mutable vs immutable types
"""

# ============================================================================
# 1. PRIMITIVE DATA TYPES
# ============================================================================

print("=" * 70)
print("1. PRIMITIVE DATA TYPES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Integer (int)
# ----------------------------------------------------------------------------
print("\n--- Integer (int) ---")

# Integers are whole numbers (positive, negative, or zero)
age = 25
negative_number = -42
zero = 0
large_number = 1000000

# Python 3 has unlimited precision for integers
very_large = 10**100  # 10 to the power of 100

print(f"age = {age}, type = {type(age)}")
print(f"negative_number = {negative_number}, type = {type(negative_number)}")
print(f"very_large = {very_large}")

# Integer operations
result = 10 + 5
print(f"10 + 5 = {result}")

# Integer division (floor division)
floor_div = 10 // 3  # Returns 3 (not 3.333...)
print(f"10 // 3 = {floor_div}")

# Modulo (remainder)
remainder = 10 % 3  # Returns 1
print(f"10 % 3 = {remainder}")

# Exponentiation
power = 2 ** 8  # 2 to the power of 8 = 256
print(f"2 ** 8 = {power}")


# ----------------------------------------------------------------------------
# 1.2 Float (float)
# ----------------------------------------------------------------------------
print("\n--- Float (float) ---")

# Floats are decimal numbers
price = 19.99
temperature = -5.5
pi = 3.14159
scientific = 1.5e3  # 1.5 * 10^3 = 1500.0

print(f"price = {price}, type = {type(price)}")
print(f"scientific = {scientific}, type = {type(scientific)}")

# Float precision issues (important for interviews!)
# Floating point arithmetic can have precision errors
result = 0.1 + 0.2
print(f"0.1 + 0.2 = {result}")  # Might not be exactly 0.3
print(f"0.1 + 0.2 == 0.3: {0.1 + 0.2 == 0.3}")  # False!

# Better way to compare floats
import math
print(f"math.isclose(0.1 + 0.2, 0.3): {math.isclose(0.1 + 0.2, 0.3)}")

# Special float values
infinity = float('inf')
negative_infinity = float('-inf')
not_a_number = float('nan')

print(f"infinity = {infinity}")
print(f"negative_infinity = {negative_infinity}")
print(f"not_a_number = {not_a_number}")
print(f"math.isnan(not_a_number): {math.isnan(not_a_number)}")


# ----------------------------------------------------------------------------
# 1.3 String (str)
# ----------------------------------------------------------------------------
print("\n--- String (str) ---")

# Strings are sequences of characters
name = "Alice"
greeting = 'Hello'
multiline = """This is a
multiline string"""

print(f"name = {name}, type = {type(name)}")
print(f"greeting = {greeting}")

# String concatenation
full_greeting = greeting + ", " + name + "!"
print(f"full_greeting = {full_greeting}")

# String repetition
repeated = "ha" * 3  # "hahaha"
print(f"'ha' * 3 = {repeated}")

# String indexing (0-based)
text = "Python"
print(f"text[0] = {text[0]}")  # 'P'
print(f"text[-1] = {text[-1]}")  # 'n' (last character)
print(f"text[1:4] = {text[1:4]}")  # 'yth' (slicing)

# String methods (immutable - returns new string)
upper_text = text.upper()  # Doesn't modify original
print(f"text.upper() = {upper_text}, original text = {text}")

# f-strings (Python 3.6+, preferred method)
age = 25
message = f"My name is {name} and I am {age} years old"
print(f"message = {message}")

# String formatting
formatted = "Value: {:.2f}".format(3.14159)  # "Value: 3.14"
print(f"formatted = {formatted}")


# ----------------------------------------------------------------------------
# 1.4 Boolean (bool)
# ----------------------------------------------------------------------------
print("\n--- Boolean (bool) ---")

# Booleans are True or False (capitalized!)
is_active = True
is_complete = False

print(f"is_active = {is_active}, type = {type(is_active)}")
print(f"is_complete = {is_complete}")

# Boolean values are actually integers (True = 1, False = 0)
print(f"True + True = {True + True}")  # 2
print(f"False + False = {False + False}")  # 0
print(f"True * 5 = {True * 5}")  # 5

# Truthy and Falsy values
# Falsy values: False, None, 0, 0.0, "", [], {}, set()
# Everything else is truthy

print(f"bool(0) = {bool(0)}")  # False
print(f"bool(1) = {bool(1)}")  # True
print(f"bool('') = {bool('')}")  # False
print(f"bool('hello') = {bool('hello')}")  # True
print(f"bool([]) = {bool([])}")  # False
print(f"bool([1, 2]) = {bool([1, 2])}")  # True


# ----------------------------------------------------------------------------
# 1.5 None (NoneType)
# ----------------------------------------------------------------------------
print("\n--- None (NoneType) ---")

# None represents the absence of a value
result = None
print(f"result = {result}, type = {type(result)}")

# None is falsy
print(f"bool(None) = {bool(None)}")  # False

# Common use case: default return value or placeholder
def find_item(items, target):
    for item in items:
        if item == target:
            return item
    return None  # Not found

# Checking for None
value = None
if value is None:  # Use 'is' not '==' for None
    print("Value is None")

# None vs empty string/list
empty_string = ""
empty_list = []
none_value = None

print(f"empty_string is None: {empty_string is None}")  # False
print(f"empty_list is None: {empty_list is None}")  # False
print(f"none_value is None: {none_value is None}")  # True


# ============================================================================
# 2. TYPE CONVERSION
# ============================================================================

print("\n" + "=" * 70)
print("2. TYPE CONVERSION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 int() - Convert to Integer
# ----------------------------------------------------------------------------
print("\n--- int() - Convert to Integer ---")

# From string
num_str = "42"
num_int = int(num_str)
print(f"int('42') = {num_int}, type = {type(num_int)}")

# From float (truncates, doesn't round)
float_num = 3.7
int_from_float = int(float_num)
print(f"int(3.7) = {int_from_float}")  # 3 (not 4!)

# From boolean
print(f"int(True) = {int(True)}")  # 1
print(f"int(False) = {int(False)}")  # 0

# Base conversion (binary, octal, hexadecimal)
binary_str = "1010"
decimal = int(binary_str, 2)  # Convert from base 2
print(f"int('1010', 2) = {decimal}")  # 10

hex_str = "FF"
decimal_from_hex = int(hex_str, 16)
print(f"int('FF', 16) = {decimal_from_hex}")  # 255

# Error cases
try:
    invalid = int("hello")  # ValueError
except ValueError as e:
    print(f"Error converting 'hello' to int: {e}")


# ----------------------------------------------------------------------------
# 2.2 float() - Convert to Float
# ----------------------------------------------------------------------------
print("\n--- float() - Convert to Float ---")

# From string
float_str = "3.14"
float_num = float(float_str)
print(f"float('3.14') = {float_num}")

# From integer
int_num = 5
float_from_int = float(int_num)
print(f"float(5) = {float_from_int}")  # 5.0

# From boolean
print(f"float(True) = {float(True)}")  # 1.0
print(f"float(False) = {float(False)}")  # 0.0

# Scientific notation
scientific = float("1.5e2")
print(f"float('1.5e2') = {scientific}")  # 150.0


# ----------------------------------------------------------------------------
# 2.3 str() - Convert to String
# ----------------------------------------------------------------------------
print("\n--- str() - Convert to String ---")

# From integer
num = 42
num_str = str(num)
print(f"str(42) = '{num_str}', type = {type(num_str)}")

# From float
pi = 3.14159
pi_str = str(pi)
print(f"str(3.14159) = '{pi_str}'")

# From boolean
bool_str = str(True)
print(f"str(True) = '{bool_str}'")  # 'True'

# From None
none_str = str(None)
print(f"str(None) = '{none_str}'")  # 'None'

# From list (and other objects)
my_list = [1, 2, 3]
list_str = str(my_list)
print(f"str([1, 2, 3]) = '{list_str}'")  # '[1, 2, 3]'


# ----------------------------------------------------------------------------
# 2.4 bool() - Convert to Boolean
# ----------------------------------------------------------------------------
print("\n--- bool() - Convert to Boolean ---")

# From integer
print(f"bool(0) = {bool(0)}")  # False
print(f"bool(1) = {bool(1)}")  # True
print(f"bool(-1) = {bool(-1)}")  # True (any non-zero is True)

# From float
print(f"bool(0.0) = {bool(0.0)}")  # False
print(f"bool(0.1) = {bool(0.1)}")  # True

# From string
print(f"bool('') = {bool('')}")  # False (empty string)
print(f"bool('hello') = {bool('hello')}")  # True (non-empty)

# From None
print(f"bool(None) = {bool(None)}")  # False

# From collections
print(f"bool([]) = {bool([])}")  # False (empty list)
print(f"bool([1]) = {bool([1])}")  # True (non-empty)
print(f"bool({{}}) = {bool({})}")  # False (empty dict)
print(f"bool({{'key': 'value'}}) = {bool({'key': 'value'})}")  # True


# ============================================================================
# 3. TYPE CHECKING
# ============================================================================

print("\n" + "=" * 70)
print("3. TYPE CHECKING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 type() - Get Type of Object
# ----------------------------------------------------------------------------
print("\n--- type() - Get Type of Object ---")

x = 42
y = 3.14
z = "hello"
w = True

print(f"type(42) = {type(x)}")
print(f"type(3.14) = {type(y)}")
print(f"type('hello') = {type(z)}")
print(f"type(True) = {type(w)}")

# type() returns the type object
print(f"type(42) == int: {type(x) == int}")  # True
print(f"type(3.14) == float: {type(y) == float}")  # True

# Getting type name as string
print(f"type(42).__name__ = {type(x).__name__}")  # 'int'


# ----------------------------------------------------------------------------
# 3.2 isinstance() - Check if Object is Instance of Type
# ----------------------------------------------------------------------------
print("\n--- isinstance() - Check if Object is Instance of Type ---")

# isinstance() is preferred over type() == for type checking
value = 42

print(f"isinstance(42, int) = {isinstance(value, int)}")  # True
print(f"isinstance(42, float) = {isinstance(value, float)}")  # False
print(f"isinstance(42, (int, float)) = {isinstance(value, (int, float))}")  # True

# isinstance() works with inheritance (important for OOP)
class Animal:
    pass

class Dog(Animal):
    pass

my_dog = Dog()
print(f"isinstance(my_dog, Dog) = {isinstance(my_dog, Dog)}")  # True
print(f"isinstance(my_dog, Animal) = {isinstance(my_dog, Animal)}")  # True (inheritance)

# type() doesn't handle inheritance the same way
print(f"type(my_dog) == Animal: {type(my_dog) == Animal}")  # False
print(f"type(my_dog) == Dog: {type(my_dog) == Dog}")  # True

# Best practice: Use isinstance() for type checking
def process_number(num):
    if isinstance(num, (int, float)):
        return num * 2
    else:
        raise TypeError(f"Expected int or float, got {type(num)}")

print(f"process_number(5) = {process_number(5)}")  # 10
print(f"process_number(3.5) = {process_number(3.5)}")  # 7.0


# ============================================================================
# 4. IMMUTABILITY
# ============================================================================

print("\n" + "=" * 70)
print("4. IMMUTABILITY")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Immutable Types
# ----------------------------------------------------------------------------
print("\n--- Immutable Types ---")

# Immutable types: int, float, str, bool, tuple, frozenset, None
# Once created, their value cannot be changed

# Integers are immutable
x = 5
print(f"x = {x}, id(x) = {id(x)}")
x = x + 1  # Creates a NEW integer object
print(f"x = {x}, id(x) = {id(x)}")  # Different id!

# Strings are immutable
text = "Hello"
print(f"text = {text}, id(text) = {id(text)}")
text = text + " World"  # Creates a NEW string object
print(f"text = {text}, id(text) = {id(text)}")  # Different id!

# Tuples are immutable
my_tuple = (1, 2, 3)
print(f"my_tuple = {my_tuple}, id(my_tuple) = {id(my_tuple)}")
# my_tuple[0] = 10  # This would raise TypeError!

# Trying to modify immutable types
try:
    text = "Hello"
    text[0] = "h"  # TypeError: 'str' object does not support item assignment
except TypeError as e:
    print(f"Cannot modify string: {e}")


# ----------------------------------------------------------------------------
# 4.2 Mutable Types
# ----------------------------------------------------------------------------
print("\n--- Mutable Types ---")

# Mutable types: list, dict, set
# Their value can be changed after creation

# Lists are mutable
my_list = [1, 2, 3]
print(f"my_list = {my_list}, id(my_list) = {id(my_list)}")
my_list[0] = 10  # Modifies the list in place
print(f"my_list = {my_list}, id(my_list) = {id(my_list)}")  # Same id!

# Dictionaries are mutable
my_dict = {"a": 1, "b": 2}
print(f"my_dict = {my_dict}, id(my_dict) = {id(my_dict)}")
my_dict["c"] = 3  # Modifies the dict in place
print(f"my_dict = {my_dict}, id(my_dict) = {id(my_dict)}")  # Same id!

# Sets are mutable
my_set = {1, 2, 3}
print(f"my_set = {my_set}, id(my_set) = {id(my_set)}")
my_set.add(4)  # Modifies the set in place
print(f"my_set = {my_set}, id(my_set) = {id(my_set)}")  # Same id!


# ----------------------------------------------------------------------------
# 4.3 Implications of Immutability
# ----------------------------------------------------------------------------
print("\n--- Implications of Immutability ---")

# 1. String operations create new strings (memory consideration)
text = "Hello"
for i in range(1000):
    text = text + " World"  # Creates 1000 new strings! Inefficient!
# Better: use list and join()
parts = ["Hello"]
for i in range(1000):
    parts.append(" World")
efficient_text = "".join(parts)  # Creates only one final string

# 2. Tuple unpacking works because tuples are immutable
point = (3, 4)
x, y = point  # Unpacking
print(f"Unpacked: x={x}, y={y}")

# 3. Dictionary keys must be immutable
valid_key = (1, 2)  # Tuple is immutable, can be a key
# invalid_key = [1, 2]  # List is mutable, cannot be a key
my_dict = {valid_key: "value"}
print(f"Dictionary with tuple key: {my_dict}")

# 4. Function arguments - immutable types are passed by value (conceptually)
def modify_immutable(x):
    x = x + 1  # Creates new object, doesn't modify original
    return x

num = 5
result = modify_immutable(num)
print(f"num = {num}, result = {result}")  # num unchanged

# Mutable types are passed by reference (conceptually)
def modify_mutable(lst):
    lst.append(4)  # Modifies the original list
    return lst

my_list = [1, 2, 3]
modify_mutable(my_list)
print(f"my_list after function call: {my_list}")  # [1, 2, 3, 4]


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Type conversion
print("\n--- Exercise 1: Type Conversion ---")
# Convert the string "123" to an integer, then to a float, then back to string
def exercise1():
    value = "123"
    as_int = int(value)
    as_float = float(value)
    back_to_str = str(as_float)
    return as_int, as_float, back_to_str

result = exercise1()
print(f"Exercise 1 result: {result}")

# Exercise 2: Type checking
print("\n--- Exercise 2: Type Checking ---")
# Write a function that accepts any type and returns its type name
def get_type_name(value):
    return type(value).__name__

print(f"get_type_name(42) = {get_type_name(42)}")
print(f"get_type_name('hello') = {get_type_name('hello')}")
print(f"get_type_name([1, 2, 3]) = {get_type_name([1, 2, 3])}")

# Exercise 3: Immutability demonstration
print("\n--- Exercise 3: Immutability ---")
# Show that strings are immutable but lists are mutable
def exercise3():
    # String (immutable)
    s = "Hello"
    s_id_before = id(s)
    s = s + " World"  # Creates new string
    s_id_after = id(s)
    string_immutable = s_id_before != s_id_after
    
    # List (mutable)
    lst = [1, 2, 3]
    lst_id_before = id(lst)
    lst.append(4)  # Modifies in place
    lst_id_after = id(lst)
    list_mutable = lst_id_before == lst_id_after
    
    return string_immutable, list_mutable

result = exercise3()
print(f"String is immutable: {result[0]}, List is mutable: {result[1]}")

# Exercise 4: Truthy/Falsy values
print("\n--- Exercise 4: Truthy/Falsy Values ---")
# Write a function that returns True if value is truthy, False otherwise
def is_truthy(value):
    return bool(value)

test_values = [0, 1, "", "hello", [], [1], None, True, False]
for val in test_values:
    print(f"is_truthy({repr(val)}) = {is_truthy(val)}")


# ============================================================================
# 6. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between == and is?
print("\n--- Q1: == vs is ---")
# == compares values, is compares object identity
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(f"a == b: {a == b}")  # True (same values)
print(f"a is b: {a is b}")  # False (different objects)
print(f"a is c: {a is c}")  # True (same object)

# For immutable types, Python may reuse objects
x = 256
y = 256
print(f"x is y (small ints): {x is y}")  # True (Python caches small integers)

x = 257
y = 257
print(f"x is y (large ints): {x is y}")  # May be False (implementation dependent)

# Q2: Why can't we use lists as dictionary keys?
print("\n--- Q2: Lists as Dictionary Keys ---")
# Lists are mutable, so their hash could change
# Dictionary keys must be hashable (immutable)
try:
    my_dict = {[1, 2]: "value"}  # TypeError
except TypeError as e:
    print(f"Cannot use list as key: {e}")

# But tuples (immutable) can be keys
my_dict = {(1, 2): "value"}
print(f"Tuple as key works: {my_dict}")

# Q3: What happens when you modify a mutable object passed to a function?
print("\n--- Q3: Mutable Objects in Functions ---")
def add_item(lst, item):
    lst.append(item)  # Modifies the original list
    return lst

original = [1, 2, 3]
result = add_item(original, 4)
print(f"original = {original}")  # Modified!
print(f"result = {result}")
print(f"original is result: {original is result}")  # Same object

# Q4: How do you check if a variable is None?
print("\n--- Q4: Checking for None ---")
value = None

# Correct way
if value is None:
    print("Value is None (correct)")

# Also works but less preferred
if value == None:
    print("Value is None (works but not preferred)")

# Wrong way (doesn't work)
if not value:
    print("This prints for None, but also for False, 0, '', [], etc.")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. IMMUTABLE TYPES: int, float, str, bool, tuple, frozenset, None
   - Cannot be modified after creation
   - Operations create new objects
   - Can be used as dictionary keys

2. MUTABLE TYPES: list, dict, set
   - Can be modified in place
   - Operations modify the same object
   - Cannot be used as dictionary keys

3. TYPE CHECKING:
   - Use isinstance() instead of type() == for type checking
   - isinstance() handles inheritance correctly

4. TYPE CONVERSION:
   - int() truncates floats (doesn't round)
   - str() can convert any object to string
   - bool() returns False for: False, None, 0, 0.0, "", [], {}, set()

5. NONE:
   - Use 'is None' not '== None' to check for None
   - None is falsy but not equal to False

6. STRING OPERATIONS:
   - Strings are immutable, so concatenation creates new strings
   - Use join() for efficient string building from multiple parts

7. == vs is:
   - == compares values
   - is compares object identity
   - Use is for None, True, False
   - Use == for value comparison
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
