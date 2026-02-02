"""
Python Operators & Expressions - Interview Preparation
Topic 1.2: Operators & Expressions

This module covers:
- Arithmetic: +, -, *, /, //, %, **
- Comparison: ==, !=, <, >, <=, >=, is, is not
- Logical: and, or, not
- Membership: in, not in
- Bitwise: &, |, ^, ~, <<, >>
- Assignment: =, +=, -=, *=, /=, //=, %=, **=
"""

# ============================================================================
# 1. ARITHMETIC OPERATORS
# ============================================================================

print("=" * 70)
print("1. ARITHMETIC OPERATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Addition (+)
# ----------------------------------------------------------------------------
print("\n--- Addition (+) ---")

# Numbers
result = 5 + 3
print(f"5 + 3 = {result}")

# Floats
result = 3.5 + 2.7
print(f"3.5 + 2.7 = {result}")

# Strings (concatenation)
greeting = "Hello" + " " + "World"
print(f"'Hello' + ' ' + 'World' = '{greeting}'")

# Lists (concatenation)
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2
print(f"[1, 2, 3] + [4, 5, 6] = {combined}")

# Tuples
tuple1 = (1, 2)
tuple2 = (3, 4)
combined_tuple = tuple1 + tuple2
print(f"(1, 2) + (3, 4) = {combined_tuple}")


# ----------------------------------------------------------------------------
# 1.2 Subtraction (-)
# ----------------------------------------------------------------------------
print("\n--- Subtraction (-) ---")

# Numbers
result = 10 - 4
print(f"10 - 4 = {result}")

# Floats
result = 7.5 - 2.3
print(f"7.5 - 2.3 = {result}")

# Negative numbers
result = 5 - 10
print(f"5 - 10 = {result}")

# Sets (difference)
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
difference = set1 - set2  # Elements in set1 but not in set2
print(f"{{1, 2, 3, 4}} - {{3, 4, 5, 6}} = {difference}")


# ----------------------------------------------------------------------------
# 1.3 Multiplication (*)
# ----------------------------------------------------------------------------
print("\n--- Multiplication (*) ---")

# Numbers
result = 6 * 7
print(f"6 * 7 = {result}")

# Floats
result = 2.5 * 4
print(f"2.5 * 4 = {result}")

# Strings (repetition)
repeated = "ha" * 3
print(f"'ha' * 3 = '{repeated}'")

# Lists (repetition)
repeated_list = [1, 2] * 3
print(f"[1, 2] * 3 = {repeated_list}")

# Tuples (repetition)
repeated_tuple = (1, 2) * 2
print(f"(1, 2) * 2 = {repeated_tuple}")


# ----------------------------------------------------------------------------
# 1.4 Division (/)
# ----------------------------------------------------------------------------
print("\n--- Division (/) ---")

# Regular division always returns float
result = 10 / 2
print(f"10 / 2 = {result}, type = {type(result)}")

result = 7 / 3
print(f"7 / 3 = {result}")  # 2.3333333333333335

# Even dividing integers returns float
result = 8 / 4
print(f"8 / 4 = {result}, type = {type(result)}")  # 2.0, not 2

# Division by zero
try:
    result = 10 / 0  # ZeroDivisionError
except ZeroDivisionError as e:
    print(f"Division by zero error: {e}")


# ----------------------------------------------------------------------------
# 1.5 Floor Division (//)
# ----------------------------------------------------------------------------
print("\n--- Floor Division (//) ---")

# Returns the largest integer less than or equal to the result
result = 10 // 3
print(f"10 // 3 = {result}")  # 3 (not 3.333...)

result = 10 // 2
print(f"10 // 2 = {result}")  # 5

# With floats
result = 10.5 // 3
print(f"10.5 // 3 = {result}")  # 3.0 (float result)

result = -10 // 3
print(f"-10 // 3 = {result}")  # -4 (rounds down, not toward zero)

# Floor division by zero
try:
    result = 10 // 0  # ZeroDivisionError
except ZeroDivisionError as e:
    print(f"Floor division by zero error: {e}")


# ----------------------------------------------------------------------------
# 1.6 Modulo (%)
# ----------------------------------------------------------------------------
print("\n--- Modulo (%) ---")

# Returns the remainder after division
result = 10 % 3
print(f"10 % 3 = {result}")  # 1

result = 10 % 2
print(f"10 % 2 = {result}")  # 0 (even number)

result = 15 % 4
print(f"15 % 4 = {result}")  # 3

# Common use case: Check if number is even or odd
number = 7
is_even = (number % 2 == 0)
print(f"{number} is even: {is_even}")

# String formatting (old style, still used)
name = "Alice"
age = 25
message = "My name is %s and I am %d years old" % (name, age)
print(f"String formatting: {message}")

# Modulo by zero
try:
    result = 10 % 0  # ZeroDivisionError
except ZeroDivisionError as e:
    print(f"Modulo by zero error: {e}")


# ----------------------------------------------------------------------------
# 1.7 Exponentiation (**)
# ----------------------------------------------------------------------------
print("\n--- Exponentiation (**) ---")

# Power operation
result = 2 ** 3
print(f"2 ** 3 = {result}")  # 8

result = 5 ** 2
print(f"5 ** 2 = {result}")  # 25

# Square root (using fractional exponent)
import math
result = 16 ** 0.5
print(f"16 ** 0.5 = {result}")  # 4.0

# Cube root
result = 27 ** (1/3)
print(f"27 ** (1/3) = {result}")  # 3.0

# Large numbers
result = 10 ** 100
print(f"10 ** 100 = {result}")  # Very large number


# ============================================================================
# 2. COMPARISON OPERATORS
# ============================================================================

print("\n" + "=" * 70)
print("2. COMPARISON OPERATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Equality (==)
# ----------------------------------------------------------------------------
print("\n--- Equality (==) ---")

# Numbers
print(f"5 == 5: {5 == 5}")  # True
print(f"5 == 3: {5 == 3}")  # False

# Floats (be careful with precision!)
print(f"0.1 + 0.2 == 0.3: {0.1 + 0.2 == 0.3}")  # False!
print(f"0.1 + 0.2: {0.1 + 0.2}")  # 0.30000000000000004

# Strings
print(f"'hello' == 'hello': {'hello' == 'hello'}")  # True
print(f"'hello' == 'world': {'hello' == 'world'}")  # False

# Lists
print(f"[1, 2, 3] == [1, 2, 3]: {[1, 2, 3] == [1, 2, 3]}")  # True
print(f"[1, 2, 3] == [1, 2]: {[1, 2, 3] == [1, 2]}")  # False

# Different types
print(f"5 == '5': {5 == '5'}")  # False (different types)


# ----------------------------------------------------------------------------
# 2.2 Inequality (!=)
# ----------------------------------------------------------------------------
print("\n--- Inequality (!=) ---")

print(f"5 != 3: {5 != 3}")  # True
print(f"5 != 5: {5 != 5}")  # False

print(f"'hello' != 'world': {'hello' != 'world'}")  # True
print(f"'hello' != 'hello': {'hello' != 'hello'}")  # False


# ----------------------------------------------------------------------------
# 2.3 Less Than (<)
# ----------------------------------------------------------------------------
print("\n--- Less Than (<) ---")

print(f"3 < 5: {3 < 5}")  # True
print(f"5 < 3: {5 < 3}")  # False
print(f"5 < 5: {5 < 5}")  # False

# Strings (lexicographic order)
print(f"'apple' < 'banana': {'apple' < 'banana'}")  # True
print(f"'zebra' < 'apple': {'zebra' < 'apple'}")  # False

# Lists (element-wise comparison)
print(f"[1, 2] < [1, 3]: {[1, 2] < [1, 3]}")  # True
print(f"[1, 2] < [1, 2, 3]: {[1, 2] < [1, 2, 3]}")  # True (shorter is less)


# ----------------------------------------------------------------------------
# 2.4 Greater Than (>)
# ----------------------------------------------------------------------------
print("\n--- Greater Than (>) ---")

print(f"5 > 3: {5 > 3}")  # True
print(f"3 > 5: {3 > 5}")  # False
print(f"5 > 5: {5 > 5}")  # False

# Strings
print(f"'banana' > 'apple': {'banana' > 'apple'}")  # True


# ----------------------------------------------------------------------------
# 2.5 Less Than or Equal (<=)
# ----------------------------------------------------------------------------
print("\n--- Less Than or Equal (<=) ---")

print(f"3 <= 5: {3 <= 5}")  # True
print(f"5 <= 5: {5 <= 5}")  # True
print(f"5 <= 3: {5 <= 3}")  # False


# ----------------------------------------------------------------------------
# 2.6 Greater Than or Equal (>=)
# ----------------------------------------------------------------------------
print("\n--- Greater Than or Equal (>=) ---")

print(f"5 >= 3: {5 >= 3}")  # True
print(f"5 >= 5: {5 >= 5}")  # True
print(f"3 >= 5: {3 >= 5}")  # False


# ----------------------------------------------------------------------------
# 2.7 Identity (is) and (is not)
# ----------------------------------------------------------------------------
print("\n--- Identity (is) and (is not) ---")

# 'is' checks if two variables refer to the SAME object
# '==' checks if two objects have the SAME VALUE

# Lists (mutable)
list1 = [1, 2, 3]
list2 = [1, 2, 3]
list3 = list1

print(f"list1 == list2: {list1 == list2}")  # True (same values)
print(f"list1 is list2: {list1 is list2}")  # False (different objects)
print(f"list1 is list3: {list1 is list3}")  # True (same object)

# Integers (immutable - Python may cache small integers)
x = 256
y = 256
print(f"x is y (256): {x is y}")  # True (cached)

x = 257
y = 257
print(f"x is y (257): {x is y}")  # May be False (implementation dependent)

# Strings (immutable - Python may intern them)
s1 = "hello"
s2 = "hello"
print(f"s1 is s2: {s1 is s2}")  # Usually True (string interning)

# None (always use 'is' for None)
value = None
print(f"value is None: {value is None}")  # True
print(f"value == None: {value == None}")  # True but not recommended

# Boolean (always use 'is' for True/False)
flag = True
print(f"flag is True: {flag is True}")  # True

# 'is not' is the negation of 'is'
print(f"list1 is not list2: {list1 is not list2}")  # True


# ============================================================================
# 3. LOGICAL OPERATORS
# ============================================================================

print("\n" + "=" * 70)
print("3. LOGICAL OPERATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 and
# ----------------------------------------------------------------------------
print("\n--- Logical AND (and) ---")

# Returns the first falsy value, or the last value if all are truthy
print(f"True and True: {True and True}")  # True
print(f"True and False: {True and False}")  # False
print(f"False and True: {False and True}")  # False
print(f"False and False: {False and False}")  # False

# Short-circuit evaluation
print(f"True and 'hello': {True and 'hello'}")  # 'hello' (last truthy)
print(f"False and 'hello': {False and 'hello'}")  # False (short-circuits)

# Common pattern: Check multiple conditions
age = 25
has_license = True
can_drive = age >= 18 and has_license
print(f"Can drive (age={age}, has_license={has_license}): {can_drive}")

# Returns first falsy value
print(f"5 and 0 and 10: {5 and 0 and 10}")  # 0 (first falsy)
print(f"5 and 10 and 20: {5 and 10 and 20}")  # 20 (all truthy, returns last)


# ----------------------------------------------------------------------------
# 3.2 or
# ----------------------------------------------------------------------------
print("\n--- Logical OR (or) ---")

# Returns the first truthy value, or the last value if all are falsy
print(f"True or True: {True or True}")  # True
print(f"True or False: {True or False}")  # True
print(f"False or True: {False or True}")  # True
print(f"False or False: {False or False}")  # False

# Short-circuit evaluation
print(f"True or 'hello': {True or 'hello'}")  # True (short-circuits)
print(f"False or 'hello': {False or 'hello'}")  # 'hello' (first truthy)

# Common pattern: Default values
name = None
display_name = name or "Guest"
print(f"Display name: {display_name}")  # "Guest"

# Returns first truthy value
print(f"0 or '' or 'hello': {0 or '' or 'hello'}")  # 'hello' (first truthy)
print(f"0 or '' or []: {0 or '' or []}")  # [] (all falsy, returns last)


# ----------------------------------------------------------------------------
# 3.3 not
# ----------------------------------------------------------------------------
print("\n--- Logical NOT (not) ---")

# Returns the opposite boolean value
print(f"not True: {not True}")  # False
print(f"not False: {not False}")  # True

# Works with truthy/falsy values
print(f"not 0: {not 0}")  # True
print(f"not 1: {not 1}")  # False
print(f"not '': {not ''}")  # True
print(f"not 'hello': {not 'hello'}")  # False
print(f"not []: {not []}")  # True
print(f"not [1, 2]: {not [1, 2]}")  # False

# Common pattern: Check if empty
items = []
if not items:
    print("List is empty")

# Double negation
print(f"not not 5: {not not 5}")  # True (converts to bool)


# ----------------------------------------------------------------------------
# 3.4 Operator Precedence
# ----------------------------------------------------------------------------
print("\n--- Operator Precedence ---")

# Order: not > and > or
result = True or False and False
print(f"True or False and False = {result}")  # True (and evaluated first)

result = not True and False
print(f"not True and False = {result}")  # False (not evaluated first)

# Use parentheses for clarity
result = (True or False) and False
print(f"(True or False) and False = {result}")  # False


# ============================================================================
# 4. MEMBERSHIP OPERATORS
# ============================================================================

print("\n" + "=" * 70)
print("4. MEMBERSHIP OPERATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 in
# ----------------------------------------------------------------------------
print("\n--- Membership (in) ---")

# Strings
text = "hello"
print(f"'e' in 'hello': {'e' in text}")  # True
print(f"'x' in 'hello': {'x' in text}")  # False
print(f"'ll' in 'hello': {'ll' in text}")  # True (substring)

# Lists
numbers = [1, 2, 3, 4, 5]
print(f"3 in [1, 2, 3, 4, 5]: {3 in numbers}")  # True
print(f"10 in [1, 2, 3, 4, 5]: {10 in numbers}")  # False

# Tuples
point = (1, 2, 3)
print(f"2 in (1, 2, 3): {2 in point}")  # True

# Dictionaries (checks keys, not values)
my_dict = {"a": 1, "b": 2, "c": 3}
print(f"'a' in {{'a': 1, 'b': 2}}: {'a' in my_dict}")  # True (key)
print(f"1 in {{'a': 1, 'b': 2}}: {1 in my_dict}")  # False (value, not key)

# Sets
my_set = {1, 2, 3, 4, 5}
print(f"3 in {{1, 2, 3, 4, 5}}: {3 in my_set}")  # True
print(f"10 in {{1, 2, 3, 4, 5}}: {10 in my_set}")  # False

# Range
print(f"5 in range(10): {5 in range(10)}")  # True
print(f"15 in range(10): {15 in range(10)}")  # False


# ----------------------------------------------------------------------------
# 4.2 not in
# ----------------------------------------------------------------------------
print("\n--- Membership (not in) ---")

text = "hello"
print(f"'x' not in 'hello': {'x' not in text}")  # True
print(f"'e' not in 'hello': {'e' not in text}")  # False

numbers = [1, 2, 3, 4, 5]
print(f"10 not in [1, 2, 3, 4, 5]: {10 not in numbers}")  # True


# ============================================================================
# 5. BITWISE OPERATORS
# ============================================================================

print("\n" + "=" * 70)
print("5. BITWISE OPERATORS")
print("=" * 70)

# Note: Bitwise operators work on binary representations of integers

# ----------------------------------------------------------------------------
# 5.1 AND (&)
# ----------------------------------------------------------------------------
print("\n--- Bitwise AND (&) ---")

# Performs AND operation on each bit
# 5 = 101 (binary)
# 3 = 011 (binary)
# 5 & 3 = 001 (binary) = 1
result = 5 & 3
print(f"5 & 3 = {result}")  # 1
print(f"Binary: 101 & 011 = 001")

# Another example
# 12 = 1100 (binary)
# 10 = 1010 (binary)
# 12 & 10 = 1000 (binary) = 8
result = 12 & 10
print(f"12 & 10 = {result}")  # 8


# ----------------------------------------------------------------------------
# 5.2 OR (|)
# ----------------------------------------------------------------------------
print("\n--- Bitwise OR (|) ---")

# Performs OR operation on each bit
# 5 = 101 (binary)
# 3 = 011 (binary)
# 5 | 3 = 111 (binary) = 7
result = 5 | 3
print(f"5 | 3 = {result}")  # 7
print(f"Binary: 101 | 011 = 111")

# Another example
result = 12 | 10
print(f"12 | 10 = {result}")  # 14 (1110 in binary)


# ----------------------------------------------------------------------------
# 5.3 XOR (^)
# ----------------------------------------------------------------------------
print("\n--- Bitwise XOR (^) ---")

# Performs XOR (exclusive OR) operation on each bit
# Returns 1 if bits are different, 0 if same
# 5 = 101 (binary)
# 3 = 011 (binary)
# 5 ^ 3 = 110 (binary) = 6
result = 5 ^ 3
print(f"5 ^ 3 = {result}")  # 6
print(f"Binary: 101 ^ 011 = 110")

# XOR with same number returns 0
result = 5 ^ 5
print(f"5 ^ 5 = {result}")  # 0

# Useful for swapping without temporary variable
a = 5
b = 3
a = a ^ b
b = a ^ b
a = a ^ b
print(f"After swap: a={a}, b={b}")  # a=3, b=5


# ----------------------------------------------------------------------------
# 5.4 NOT (~)
# ----------------------------------------------------------------------------
print("\n--- Bitwise NOT (~) ---")

# Flips all bits (one's complement)
# In Python, ~x = -(x+1) due to two's complement representation
result = ~5
print(f"~5 = {result}")  # -6
print(f"~0 = {~0}")  # -1
print(f"~(-1) = {~(-1)}")  # 0

# Explanation: ~5 in binary (assuming 8 bits):
# 5 = 00000101
# ~5 = 11111010 (which is -6 in two's complement)


# ----------------------------------------------------------------------------
# 5.5 Left Shift (<<)
# ----------------------------------------------------------------------------
print("\n--- Left Shift (<<) ---")

# Shifts bits to the left, filling with zeros
# Equivalent to multiplying by 2^n
# 5 = 101 (binary)
# 5 << 1 = 1010 (binary) = 10
result = 5 << 1
print(f"5 << 1 = {result}")  # 10 (5 * 2^1)

result = 5 << 2
print(f"5 << 2 = {result}")  # 20 (5 * 2^2)

result = 3 << 3
print(f"3 << 3 = {result}")  # 24 (3 * 2^3)


# ----------------------------------------------------------------------------
# 5.6 Right Shift (>>)
# ----------------------------------------------------------------------------
print("\n--- Right Shift (>>) ---")

# Shifts bits to the right
# Equivalent to integer division by 2^n
# 10 = 1010 (binary)
# 10 >> 1 = 101 (binary) = 5
result = 10 >> 1
print(f"10 >> 1 = {result}")  # 5 (10 // 2^1)

result = 10 >> 2
print(f"10 >> 2 = {result}")  # 2 (10 // 2^2)

result = 20 >> 3
print(f"20 >> 3 = {result}")  # 2 (20 // 2^3)


# ============================================================================
# 6. ASSIGNMENT OPERATORS
# ============================================================================

print("\n" + "=" * 70)
print("6. ASSIGNMENT OPERATORS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 6.1 Basic Assignment (=)
# ----------------------------------------------------------------------------
print("\n--- Basic Assignment (=) ---")

x = 10
print(f"x = 10: x = {x}")

# Multiple assignment
a = b = c = 0
print(f"a = b = c = 0: a={a}, b={b}, c={c}")

# Tuple unpacking
x, y = 5, 10
print(f"x, y = 5, 10: x={x}, y={y}")

# List unpacking
a, b, c = [1, 2, 3]
print(f"a, b, c = [1, 2, 3]: a={a}, b={b}, c={c}")


# ----------------------------------------------------------------------------
# 6.2 Addition Assignment (+=)
# ----------------------------------------------------------------------------
print("\n--- Addition Assignment (+=) ---")

x = 5
x += 3  # Equivalent to x = x + 3
print(f"x += 3: x = {x}")  # 8

# Works with strings
text = "Hello"
text += " World"
print(f"text += ' World': text = '{text}'")

# Works with lists
my_list = [1, 2]
my_list += [3, 4]  # Equivalent to my_list.extend([3, 4])
print(f"my_list += [3, 4]: my_list = {my_list}")


# ----------------------------------------------------------------------------
# 6.3 Subtraction Assignment (-=)
# ----------------------------------------------------------------------------
print("\n--- Subtraction Assignment (-=) ---")

x = 10
x -= 3  # Equivalent to x = x - 3
print(f"x -= 3: x = {x}")  # 7


# ----------------------------------------------------------------------------
# 6.4 Multiplication Assignment (*=)
# ----------------------------------------------------------------------------
print("\n--- Multiplication Assignment (*=) ---")

x = 5
x *= 3  # Equivalent to x = x * 3
print(f"x *= 3: x = {x}")  # 15

# Works with strings
text = "ha"
text *= 3
print(f"text *= 3: text = '{text}'")  # 'hahaha'


# ----------------------------------------------------------------------------
# 6.5 Division Assignment (/=)
# ----------------------------------------------------------------------------
print("\n--- Division Assignment (/=) ---")

x = 10
x /= 2  # Equivalent to x = x / 2
print(f"x /= 2: x = {x}, type = {type(x)}")  # 5.0 (always float)


# ----------------------------------------------------------------------------
# 6.6 Floor Division Assignment (//=)
# ----------------------------------------------------------------------------
print("\n--- Floor Division Assignment (//=) ---")

x = 10
x //= 3  # Equivalent to x = x // 3
print(f"x //= 3: x = {x}")  # 3


# ----------------------------------------------------------------------------
# 6.7 Modulo Assignment (%=)
# ----------------------------------------------------------------------------
print("\n--- Modulo Assignment (%=) ---")

x = 10
x %= 3  # Equivalent to x = x % 3
print(f"x %= 3: x = {x}")  # 1


# ----------------------------------------------------------------------------
# 6.8 Exponentiation Assignment (**=)
# ----------------------------------------------------------------------------
print("\n--- Exponentiation Assignment (**=) ---")

x = 2
x **= 3  # Equivalent to x = x ** 3
print(f"x **= 3: x = {x}")  # 8


# ============================================================================
# 7. OPERATOR PRECEDENCE
# ============================================================================

print("\n" + "=" * 70)
print("7. OPERATOR PRECEDENCE")
print("=" * 70)

print("""
Order of precedence (highest to lowest):

1. Parentheses: ()
2. Exponentiation: **
3. Unary operators: +x, -x, ~x, not x
4. Multiplication, Division, Floor Division, Modulo: *, /, //, %
5. Addition, Subtraction: +, -
6. Bitwise shifts: <<, >>
7. Bitwise AND: &
8. Bitwise XOR: ^
9. Bitwise OR: |
10. Comparison operators: <, <=, >, >=, ==, !=, is, is not, in, not in
11. Logical NOT: not
12. Logical AND: and
13. Logical OR: or
14. Assignment: =, +=, -=, *=, /=, //=, %=, **=, &=, |=, ^=, <<=, >>=
""")

# Examples
result = 2 + 3 * 4
print(f"2 + 3 * 4 = {result}")  # 14 (not 20)

result = (2 + 3) * 4
print(f"(2 + 3) * 4 = {result}")  # 20

result = 2 ** 3 ** 2
print(f"2 ** 3 ** 2 = {result}")  # 512 (right-associative: 2 ** (3 ** 2))

result = not True and False
print(f"not True and False = {result}")  # False (not evaluated first)


# ============================================================================
# 8. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("8. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Check if number is even or odd
print("\n--- Exercise 1: Even/Odd Check ---")
def is_even(n):
    return n % 2 == 0

print(f"is_even(4): {is_even(4)}")  # True
print(f"is_even(5): {is_even(5)}")  # False


# Exercise 2: Check if year is leap year
print("\n--- Exercise 2: Leap Year Check ---")
def is_leap_year(year):
    # Leap year if divisible by 4, but not by 100 unless also divisible by 400
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

print(f"is_leap_year(2020): {is_leap_year(2020)}")  # True
print(f"is_leap_year(1900): {is_leap_year(1900)}")  # False
print(f"is_leap_year(2000): {is_leap_year(2000)}")  # True


# Exercise 3: Swap two variables
print("\n--- Exercise 3: Swap Variables ---")
a, b = 5, 10
print(f"Before swap: a={a}, b={b}")
a, b = b, a  # Pythonic way
print(f"After swap: a={a}, b={b}")


# Exercise 4: Check if character is vowel
print("\n--- Exercise 4: Vowel Check ---")
def is_vowel(char):
    return char.lower() in 'aeiou'

print(f"is_vowel('a'): {is_vowel('a')}")  # True
print(f"is_vowel('B'): {is_vowel('B')}")  # False


# Exercise 5: Calculate power of 2
print("\n--- Exercise 5: Power of 2 ---")
def is_power_of_2(n):
    # Using bitwise: power of 2 has only one bit set
    return n > 0 and (n & (n - 1)) == 0

print(f"is_power_of_2(8): {is_power_of_2(8)}")  # True
print(f"is_power_of_2(10): {is_power_of_2(10)}")  # False


# Exercise 6: Count set bits
print("\n--- Exercise 6: Count Set Bits ---")
def count_bits(n):
    count = 0
    while n:
        count += n & 1  # Check if last bit is set
        n >>= 1  # Right shift
    return count

print(f"count_bits(5): {count_bits(5)}")  # 2 (101 has 2 ones)
print(f"count_bits(7): {count_bits(7)}")  # 3 (111 has 3 ones)


# ============================================================================
# 9. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("9. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between == and is?
print("\n--- Q1: == vs is ---")
print("""
== compares VALUES
is compares OBJECT IDENTITY

For None, True, False: always use 'is'
For value comparison: use ==
""")

# Q2: How does short-circuit evaluation work?
print("\n--- Q2: Short-circuit Evaluation ---")
def expensive_function():
    print("This won't be called!")
    return True

# Short-circuits on False
result = False and expensive_function()  # expensive_function() not called
print("Short-circuit with 'and': expensive_function() not called")

# Short-circuits on True
result = True or expensive_function()  # expensive_function() not called
print("Short-circuit with 'or': expensive_function() not called")


# Q3: What's the result of 0.1 + 0.2?
print("\n--- Q3: Floating Point Precision ---")
result = 0.1 + 0.2
print(f"0.1 + 0.2 = {result}")  # 0.30000000000000004
print(f"0.1 + 0.2 == 0.3: {0.1 + 0.2 == 0.3}")  # False!
print("Use math.isclose() for float comparison")


# Q4: How do you check membership efficiently?
print("\n--- Q4: Membership Checking ---")
print("""
For lists: O(n) - linear search
For sets/dicts: O(1) - hash lookup (much faster!)
For strings: O(n) - substring search
""")

# Example
my_list = [1, 2, 3, 4, 5]
my_set = {1, 2, 3, 4, 5}

# Both work, but set is faster for large data
print(f"3 in list: {3 in my_list}")
print(f"3 in set: {3 in my_set}")


# ============================================================================
# 10. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("10. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. ARITHMETIC OPERATORS:
   - / always returns float (even 10/2 = 5.0)
   - // is floor division (rounds down)
   - % is modulo (remainder)
   - ** is exponentiation

2. COMPARISON OPERATORS:
   - == compares values
   - is compares object identity
   - Use 'is' for None, True, False
   - Be careful with float comparison (use math.isclose())

3. LOGICAL OPERATORS:
   - and/or return the first truthy/falsy value, not just True/False
   - Short-circuit evaluation: and stops at first False, or stops at first True
   - not returns boolean

4. MEMBERSHIP OPERATORS:
   - in/not in work with strings, lists, tuples, dicts (keys), sets
   - Sets and dicts have O(1) membership check (faster!)

5. BITWISE OPERATORS:
   - & (AND), | (OR), ^ (XOR), ~ (NOT)
   - << (left shift = multiply by 2^n)
   - >> (right shift = divide by 2^n)
   - Useful for low-level operations, flags, optimization

6. ASSIGNMENT OPERATORS:
   - +=, -=, *=, /=, //=, %=, **= are shorthand
   - += works with strings and lists (concatenation)

7. OPERATOR PRECEDENCE:
   - Use parentheses for clarity
   - ** is right-associative
   - not > and > or

8. COMMON PATTERNS:
   - n % 2 == 0 (check even)
   - n & (n-1) == 0 (check power of 2)
   - value or default (default value pattern)
   - a, b = b, a (swap)
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
