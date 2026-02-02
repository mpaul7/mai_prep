"""
Python Control Flow - Interview Preparation
Topic 1.3: Control Flow

This module covers:
- Conditionals: if, elif, else
- Loops: for, while
- Loop Control: break, continue, pass
- Ternary Operator: x if condition else y
- Nested Structures: Nested loops and conditionals
"""

# ============================================================================
# 1. CONDITIONAL STATEMENTS
# ============================================================================

print("=" * 70)
print("1. CONDITIONAL STATEMENTS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 if Statement
# ----------------------------------------------------------------------------
print("\n--- if Statement ---")

age = 20
if age >= 18:
    print("You are an adult")

# Single line if (not recommended for readability)
if age >= 18: print("Adult")

# Multiple statements
if age >= 18:
    print("You are an adult")
    print("You can vote")


# ----------------------------------------------------------------------------
# 1.2 if-else Statement
# ----------------------------------------------------------------------------
print("\n--- if-else Statement ---")

age = 15
if age >= 18:
    print("You are an adult")
else:
    print("You are a minor")

# Example: Check if number is even or odd
number = 7
if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")


# ----------------------------------------------------------------------------
# 1.3 if-elif-else Statement
# ----------------------------------------------------------------------------
print("\n--- if-elif-else Statement ---")

score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Score {score} = Grade {grade}")

# Multiple elif conditions
temperature = 25
if temperature > 30:
    weather = "Hot"
elif temperature > 20:
    weather = "Warm"
elif temperature > 10:
    weather = "Cool"
else:
    weather = "Cold"

print(f"Temperature {temperature}°C is {weather}")


# ----------------------------------------------------------------------------
# 1.4 Nested Conditionals
# ----------------------------------------------------------------------------
print("\n--- Nested Conditionals ---")

age = 25
has_license = True

if age >= 18:
    if has_license:
        print("You can drive")
    else:
        print("You need a license to drive")
else:
    print("You are too young to drive")

# Flattened version (often better)
if age >= 18 and has_license:
    print("You can drive")
elif age >= 18:
    print("You need a license to drive")
else:
    print("You are too young to drive")


# ----------------------------------------------------------------------------
# 1.5 Multiple Conditions
# ----------------------------------------------------------------------------
print("\n--- Multiple Conditions ---")

age = 25
income = 50000

# Using and
if age >= 18 and income >= 30000:
    print("Eligible for loan")

# Using or
if age < 18 or age > 65:
    print("Special category")

# Using not
if not (age < 18):
    print("Not a minor")

# Complex conditions
if (age >= 18 and age <= 65) and (income >= 30000 or has_license):
    print("Complex condition met")


# ----------------------------------------------------------------------------
# 1.6 Truthy/Falsy in Conditionals
# ----------------------------------------------------------------------------
print("\n--- Truthy/Falsy in Conditionals ---")

# Empty check
items = []
if not items:  # More Pythonic than len(items) == 0
    print("List is empty")

# Non-empty check
items = [1, 2, 3]
if items:  # More Pythonic than len(items) > 0
    print("List has items")

# String check
name = ""
if name:
    print(f"Name is {name}")
else:
    print("Name is empty")

# None check
value = None
if value is None:
    print("Value is None")

# Multiple falsy checks
data = None
if not data:
    print("Data is falsy (None, empty, 0, False, etc.)")


# ============================================================================
# 2. FOR LOOPS
# ============================================================================

print("\n" + "=" * 70)
print("2. FOR LOOPS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic for Loop
# ----------------------------------------------------------------------------
print("\n--- Basic for Loop ---")

# Iterate over a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"Fruit: {fruit}")

# Iterate over a string
word = "Python"
for char in word:
    print(char, end=" ")
print()  # New line

# Iterate over a range
for i in range(5):
    print(f"Number: {i}")

# Range with start and stop
for i in range(2, 6):
    print(f"Number: {i}")

# Range with step
for i in range(0, 10, 2):
    print(f"Even number: {i}")


# ----------------------------------------------------------------------------
# 2.2 enumerate() - Get Index and Value
# ----------------------------------------------------------------------------
print("\n--- enumerate() - Get Index and Value ---")

fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"Index {index}: {fruit}")

# Start index from 1
for index, fruit in enumerate(fruits, start=1):
    print(f"Position {index}: {fruit}")


# ----------------------------------------------------------------------------
# 2.3 zip() - Iterate Multiple Sequences
# ----------------------------------------------------------------------------
print("\n--- zip() - Iterate Multiple Sequences ---")

names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Multiple sequences
cities = ["New York", "London", "Tokyo"]
for name, age, city in zip(names, ages, cities):
    print(f"{name}, {age}, lives in {city}")


# ----------------------------------------------------------------------------
# 2.4 Iterating Over Dictionaries
# ----------------------------------------------------------------------------
print("\n--- Iterating Over Dictionaries ---")

person = {"name": "Alice", "age": 25, "city": "New York"}

# Iterate over keys (default)
for key in person:
    print(f"Key: {key}")

# Iterate over keys explicitly
for key in person.keys():
    print(f"Key: {key}")

# Iterate over values
for value in person.values():
    print(f"Value: {value}")

# Iterate over key-value pairs
for key, value in person.items():
    print(f"{key}: {value}")


# ----------------------------------------------------------------------------
# 2.5 Nested for Loops
# ----------------------------------------------------------------------------
print("\n--- Nested for Loops ---")

# Multiplication table
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i} x {j} = {i * j}", end="  ")
    print()  # New line after each row

# Iterate over nested list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for row in matrix:
    for element in row:
        print(element, end=" ")
    print()


# ----------------------------------------------------------------------------
# 2.6 List Comprehensions (Alternative to for loops)
# ----------------------------------------------------------------------------
print("\n--- List Comprehensions (Alternative to for loops) ---")

# Traditional for loop
squares = []
for i in range(5):
    squares.append(i ** 2)
print(f"Squares (for loop): {squares}")

# List comprehension (more Pythonic)
squares = [i ** 2 for i in range(5)]
print(f"Squares (list comprehension): {squares}")

# With condition
even_squares = [i ** 2 for i in range(10) if i % 2 == 0]
print(f"Even squares: {even_squares}")


# ============================================================================
# 3. WHILE LOOPS
# ============================================================================

print("\n" + "=" * 70)
print("3. WHILE LOOPS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Basic while Loop
# ----------------------------------------------------------------------------
print("\n--- Basic while Loop ---")

# Countdown
count = 5
while count > 0:
    print(f"Count: {count}")
    count -= 1
print("Blast off!")

# User input loop (simulated)
attempts = 0
max_attempts = 3
password = "secret"
entered_password = "wrong"  # Simulated input

while attempts < max_attempts and entered_password != password:
    attempts += 1
    print(f"Attempt {attempts}/{max_attempts}")
    # In real code: entered_password = input("Enter password: ")
    if attempts == max_attempts:
        entered_password = password  # Simulate correct password


# ----------------------------------------------------------------------------
# 3.2 while-else
# ----------------------------------------------------------------------------
print("\n--- while-else ---")

# else executes if loop completes normally (not via break)
count = 0
while count < 3:
    print(f"Count: {count}")
    count += 1
else:
    print("Loop completed normally")

# With break (else won't execute)
count = 0
while count < 3:
    if count == 1:
        break
    print(f"Count: {count}")
    count += 1
else:
    print("This won't print")


# ----------------------------------------------------------------------------
# 3.3 Infinite Loops
# ----------------------------------------------------------------------------
print("\n--- Infinite Loops ---")

# Infinite loop with break
count = 0
while True:
    count += 1
    if count > 5:
        break
    print(f"Count: {count}")


# ============================================================================
# 4. LOOP CONTROL STATEMENTS
# ============================================================================

print("\n" + "=" * 70)
print("4. LOOP CONTROL STATEMENTS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 break - Exit Loop Early
# ----------------------------------------------------------------------------
print("\n--- break - Exit Loop Early ---")

# Find first even number
numbers = [1, 3, 5, 8, 9, 10]
for num in numbers:
    if num % 2 == 0:
        print(f"First even number: {num}")
        break

# Search in list
target = 5
found = False
for i, num in enumerate(numbers):
    if num == target:
        print(f"Found {target} at index {i}")
        found = True
        break

if not found:
    print(f"{target} not found")


# ----------------------------------------------------------------------------
# 4.2 continue - Skip Current Iteration
# ----------------------------------------------------------------------------
print("\n--- continue - Skip Current Iteration ---")

# Print only odd numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Odd numbers:")
for num in numbers:
    if num % 2 == 0:
        continue  # Skip even numbers
    print(num, end=" ")
print()

# Process only valid data
data = [1, None, 3, None, 5, None]
valid_data = []
for item in data:
    if item is None:
        continue  # Skip None values
    valid_data.append(item)
print(f"Valid data: {valid_data}")


# ----------------------------------------------------------------------------
# 4.3 pass - Placeholder
# ----------------------------------------------------------------------------
print("\n--- pass - Placeholder ---")

# pass does nothing, used as placeholder
for i in range(5):
    if i % 2 == 0:
        pass  # Do nothing for even numbers
    else:
        print(f"Odd: {i}")

# Common use: empty function/class definition
def placeholder_function():
    pass  # To be implemented later

class PlaceholderClass:
    pass  # To be implemented later


# ----------------------------------------------------------------------------
# 4.4 break and continue in Nested Loops
# ----------------------------------------------------------------------------
print("\n--- break and continue in Nested Loops ---")

# break only exits inner loop
print("Break in nested loop:")
for i in range(3):
    for j in range(3):
        if j == 1:
            break  # Only breaks inner loop
        print(f"i={i}, j={j}")

# Using flag to break outer loop
print("\nBreak outer loop with flag:")
should_break = False
for i in range(3):
    for j in range(3):
        if i == 1 and j == 1:
            should_break = True
            break
        print(f"i={i}, j={j}")
    if should_break:
        break


# ============================================================================
# 5. TERNARY OPERATOR (Conditional Expression)
# ============================================================================

print("\n" + "=" * 70)
print("5. TERNARY OPERATOR")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Basic Ternary Operator
# ----------------------------------------------------------------------------
print("\n--- Basic Ternary Operator ---")

# Syntax: value_if_true if condition else value_if_false
age = 20
status = "adult" if age >= 18 else "minor"
print(f"Age {age}: {status}")

# Traditional if-else equivalent
if age >= 18:
    status = "adult"
else:
    status = "minor"

# Multiple examples
score = 85
grade = "Pass" if score >= 60 else "Fail"
print(f"Score {score}: {grade}")

number = 7
parity = "even" if number % 2 == 0 else "odd"
print(f"Number {number} is {parity}")


# ----------------------------------------------------------------------------
# 5.2 Nested Ternary (Use Sparingly)
# ----------------------------------------------------------------------------
print("\n--- Nested Ternary (Use Sparingly) ---")

score = 85
grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
print(f"Score {score}: Grade {grade}")

# More readable version with if-elif-else
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"


# ----------------------------------------------------------------------------
# 5.3 Ternary with Function Calls
# ----------------------------------------------------------------------------
print("\n--- Ternary with Function Calls ---")

def get_max(a, b):
    return a if a > b else b

result = get_max(5, 10)
print(f"Max of 5 and 10: {result}")

# In list comprehensions
numbers = [1, 2, 3, 4, 5]
doubled_evens = [x * 2 if x % 2 == 0 else x for x in numbers]
print(f"Doubled evens: {doubled_evens}")


# ============================================================================
# 6. NESTED STRUCTURES
# ============================================================================

print("\n" + "=" * 70)
print("6. NESTED STRUCTURES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 6.1 Nested Conditionals
# ----------------------------------------------------------------------------
print("\n--- Nested Conditionals ---")

age = 25
income = 50000
has_credit = True

if age >= 18:
    if income >= 30000:
        if has_credit:
            print("Loan approved")
        else:
            print("Need credit history")
    else:
        print("Income too low")
else:
    print("Too young for loan")

# Flattened version (often clearer)
if age >= 18 and income >= 30000 and has_credit:
    print("Loan approved")
elif age >= 18 and income >= 30000:
    print("Need credit history")
elif age >= 18:
    print("Income too low")
else:
    print("Too young for loan")


# ----------------------------------------------------------------------------
# 6.2 Nested Loops
# ----------------------------------------------------------------------------
print("\n--- Nested Loops ---")

# Print multiplication table
print("Multiplication table:")
for i in range(1, 4):
    for j in range(1, 4):
        result = i * j
        print(f"{i} x {j} = {result}", end="  ")
    print()

# Find pairs that sum to target
numbers = [1, 2, 3, 4, 5]
target = 7
print(f"\nPairs that sum to {target}:")
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        if numbers[i] + numbers[j] == target:
            print(f"({numbers[i]}, {numbers[j]})")


# ----------------------------------------------------------------------------
# 6.3 Loops with Conditionals
# ----------------------------------------------------------------------------
print("\n--- Loops with Conditionals ---")

# Filter and process
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Even numbers squared:")
for num in numbers:
    if num % 2 == 0:
        print(f"{num}^2 = {num ** 2}")

# Process matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("\nMatrix elements:")
for row_idx, row in enumerate(matrix):
    for col_idx, element in enumerate(row):
        if element % 2 == 0:
            print(f"Matrix[{row_idx}][{col_idx}] = {element} (even)")


# ----------------------------------------------------------------------------
# 6.4 Complex Nested Structures
# ----------------------------------------------------------------------------
print("\n--- Complex Nested Structures ---")

# Process nested data structure
students = [
    {"name": "Alice", "grades": [85, 90, 88]},
    {"name": "Bob", "grades": [75, 80, 82]},
    {"name": "Charlie", "grades": [95, 92, 90]}
]

print("Student averages:")
for student in students:
    name = student["name"]
    grades = student["grades"]
    total = 0
    count = 0
    for grade in grades:
        total += grade
        count += 1
    average = total / count if count > 0 else 0
    print(f"{name}: {average:.2f}")


# ============================================================================
# 7. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Find maximum in list
print("\n--- Exercise 1: Find Maximum ---")
def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val

numbers = [3, 7, 2, 9, 1, 5]
print(f"Maximum of {numbers}: {find_max(numbers)}")


# Exercise 2: Count occurrences
print("\n--- Exercise 2: Count Occurrences ---")
def count_occurrences(lst, target):
    count = 0
    for item in lst:
        if item == target:
            count += 1
    return count

data = [1, 2, 3, 2, 4, 2, 5]
print(f"Count of 2 in {data}: {count_occurrences(data, 2)}")


# Exercise 3: Check if list is sorted
print("\n--- Exercise 3: Check if Sorted ---")
def is_sorted(lst):
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True

sorted_list = [1, 2, 3, 4, 5]
unsorted_list = [1, 3, 2, 4, 5]
print(f"{sorted_list} is sorted: {is_sorted(sorted_list)}")
print(f"{unsorted_list} is sorted: {is_sorted(unsorted_list)}")


# Exercise 4: Remove duplicates
print("\n--- Exercise 4: Remove Duplicates ---")
def remove_duplicates(lst):
    seen = []
    result = []
    for item in lst:
        if item not in seen:
            seen.append(item)
            result.append(item)
    return result

data = [1, 2, 2, 3, 3, 3, 4, 5, 5]
print(f"Remove duplicates from {data}: {remove_duplicates(data)}")


# Exercise 5: FizzBuzz
print("\n--- Exercise 5: FizzBuzz ---")
def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 15 == 0:
            print("FizzBuzz", end=" ")
        elif i % 3 == 0:
            print("Fizz", end=" ")
        elif i % 5 == 0:
            print("Buzz", end=" ")
        else:
            print(i, end=" ")
    print()

print("FizzBuzz for 1-20:")
fizzbuzz(20)


# Exercise 6: Prime number checker
print("\n--- Exercise 6: Prime Number Checker ---")
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

print("Prime numbers up to 20:")
for num in range(2, 21):
    if is_prime(num):
        print(num, end=" ")
print()


# Exercise 7: Reverse a string
print("\n--- Exercise 7: Reverse String ---")
def reverse_string(s):
    result = ""
    for char in s:
        result = char + result  # Prepend character
    return result

text = "Python"
print(f"Reverse of '{text}': {reverse_string(text)}")


# Exercise 8: Find common elements
print("\n--- Exercise 8: Find Common Elements ---")
def find_common(list1, list2):
    common = []
    for item in list1:
        if item in list2 and item not in common:
            common.append(item)
    return common

list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]
print(f"Common elements: {find_common(list1, list2)}")


# ============================================================================
# 8. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("8. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between break and continue?
print("\n--- Q1: break vs continue ---")
print("""
break: Exits the loop immediately
continue: Skips current iteration and continues to next
""")

# Q2: When to use for vs while?
print("\n--- Q2: for vs while ---")
print("""
for: Use when you know the number of iterations or iterating over a sequence
while: Use when condition determines when to stop (unknown iterations)
""")

# Q3: What does pass do?
print("\n--- Q3: pass statement ---")
print("""
pass: Placeholder that does nothing
Used when syntax requires a statement but no action is needed
Common in empty functions/classes or as placeholder for future code
""")

# Q4: How to break out of nested loops?
print("\n--- Q4: Breaking Nested Loops ---")
print("""
Option 1: Use a flag variable
Option 2: Use exception handling (not recommended)
Option 3: Refactor into a function and use return
""")

# Example: Using flag
should_break = False
for i in range(3):
    for j in range(3):
        if i == 1 and j == 1:
            should_break = True
            break
    if should_break:
        break

# Example: Using function
def find_pair(numbers, target):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                return (i, j)  # Return exits function
    return None


# ============================================================================
# 9. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("9. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. CONDITIONALS:
   - if-elif-else: Use elif for multiple conditions
   - Use truthy/falsy checks: if items: instead of if len(items) > 0:
   - Use 'is None' not '== None' for None checks
   - Prefer flat conditionals over deeply nested ones

2. FOR LOOPS:
   - Use enumerate() to get index and value
   - Use zip() to iterate multiple sequences
   - Use .items() for dictionary key-value pairs
   - Consider list comprehensions for simple transformations

3. WHILE LOOPS:
   - Use when iterations are unknown
   - Always ensure loop termination condition
   - while-else executes if loop completes normally (not via break)

4. LOOP CONTROL:
   - break: Exit loop immediately
   - continue: Skip to next iteration
   - pass: Placeholder (does nothing)
   - break only exits innermost loop in nested loops

5. TERNARY OPERATOR:
   - Syntax: value_if_true if condition else value_if_false
   - Use for simple conditionals, not complex nested ones
   - More readable than if-else for simple assignments

6. NESTED STRUCTURES:
   - Can nest conditionals, loops, or both
   - Keep nesting depth reasonable (2-3 levels max)
   - Consider refactoring deeply nested code into functions

7. BEST PRACTICES:
   - Use meaningful variable names
   - Keep loops simple and readable
   - Use break/continue judiciously
   - Prefer for loops over while when possible
   - Use list comprehensions for simple transformations
   - Avoid deeply nested structures (use functions instead)
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
