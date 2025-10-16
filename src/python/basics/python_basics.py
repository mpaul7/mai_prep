"""
Python Basics for Data Science Interview Preparation

This module covers fundamental Python concepts commonly tested in 
data science interviews, particularly HackerRank-style problems.

Topics covered:
1. Scalar Types (int, float, str, bool)
2. Operators (arithmetic, comparison, logical, assignment)
3. Control Flow (if/elif/else, for, while)
4. Functions (definition, parameters, return values)
5. Lists and List Operations
6. Multiple List Iteration (zip, enumerate)
7. Loop Control (break, continue, pass)
8. String Operations and Formatting
"""

import math
from typing import List, Tuple, Union, Any, Optional


class PythonBasicsDemo:
    """
    A comprehensive demonstration of Python basics for interview preparation.
    """
    
    def __init__(self):
        """Initialize the Python basics demonstration."""
        pass
    
    # ========== SCALAR TYPES ==========
    
    def demonstrate_scalar_types(self):
        """Demonstrate Python scalar types and their operations."""
        print("=== Python Scalar Types ===\n")
        
        # Integer type
        integer_val = 42
        print(f"Integer: {integer_val}, type: {type(integer_val)}")
        print(f"Integer operations: {integer_val + 8}, {integer_val * 2}, {integer_val // 5}")
        
        # Float type
        float_val = 3.14159
        print(f"Float: {float_val}, type: {type(float_val)}")
        print(f"Float operations: {float_val + 1}, {float_val * 2}, {round(float_val, 2)}")
        
        
        # String type
        string_val = "Hello, World!"
        print(f"String: '{string_val}', type: {type(string_val)}")
        print(f"String operations: length={len(string_val)}, upper='{string_val.upper()}'")
        
        # Boolean type
        bool_val = True
        print(f"Boolean: {bool_val}, type: {type(bool_val)}")
        print(f"Boolean operations: not {bool_val} = {not bool_val}")
        
        # Type conversions
        print(f"\nType conversions:")
        print(f"int('123') = {int('123')}")
        print(f"float('3.14') = {float('3.14')}")
        print(f"str(42) = '{str(42)}'")
        print(f"bool(1) = {bool(1)}, bool(0) = {bool(0)}")
        print()
    
    # ========== OPERATORS ==========
    
    def demonstrate_operators(self):
        """Demonstrate Python operators."""
        print("=== Python Operators ===\n")
        
        a, b = 10, 3
        
        # Arithmetic operators
        print("Arithmetic Operators:")
        print(f"{a} + {b} = {a + b}")
        print(f"{a} - {b} = {a - b}")
        print(f"{a} * {b} = {a * b}")
        print(f"{a} / {b} = {a / b}")
        print(f"{a} // {b} = {a // b}")  # Floor division
        print(f"{a} % {b} = {a % b}")   # Modulo
        print(f"{a} ** {b} = {a ** b}") # Exponentiation
        
        # Comparison operators
        print(f"\nComparison Operators:")
        print(f"{a} == {b} = {a == b}")
        print(f"{a} != {b} = {a != b}")
        print(f"{a} > {b} = {a > b}")
        print(f"{a} < {b} = {a < b}")
        print(f"{a} >= {b} = {a >= b}")
        print(f"{a} <= {b} = {a <= b}")
        
        # Logical operators
        x, y = True, False
        print(f"\nLogical Operators:")
        print(f"{x} and {y} = {x and y}")
        print(f"{x} or {y} = {x or y}")
        print(f"not {x} = {not x}")
        
        # Assignment operators
        print(f"\nAssignment Operators:")
        c = 5
        print(f"c = {c}")
        c += 3
        print(f"c += 3 → c = {c}")
        c *= 2
        print(f"c *= 2 → c = {c}")
        c //= 4
        print(f"c //= 4 → c = {c}")
        print()
    
    # ========== FUNCTIONS ==========
    
    def demonstrate_functions(self):
        """Demonstrate function definition and usage."""
        print("=== Python Functions ===\n")
        
        # Basic function
        def greet(name):
            """Simple function with one parameter."""
            return f"Hello, {name}!"
        
        print(f"greet('Alice') = {greet('Alice')}")
        
        # Function with multiple parameters
        def calculate_area(length, width):
            """Function with multiple parameters."""
            return length * width
        
        print(f"calculate_area(5, 3) = {calculate_area(5, 3)}")
        
        # Function with default parameters
        def power(base, exponent=2):
            """Function with default parameter."""
            return base ** exponent
        
        print(f"power(4) = {power(4)}")
        print(f"power(4, 3) = {power(4, 3)}")
        
        # Function with variable arguments
        def sum_all(*args):
            """Function with variable arguments."""
            return sum(args)
        
        print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")
        
        # Function with keyword arguments
        def create_profile(**kwargs):
            """Function with keyword arguments."""
            return kwargs
        
        profile = create_profile(name="John", age=30, city="New York")
        print(f"create_profile(name='John', age=30, city='New York') = {profile}")
        
        # Function returning multiple values
        def get_stats(numbers):
            """Function returning multiple values."""
            return min(numbers), max(numbers), sum(numbers) / len(numbers)
        
        data = [1, 2, 3, 4, 5]
        min_val, max_val, avg_val = get_stats(data)
        print(f"get_stats({data}) = min={min_val}, max={max_val}, avg={avg_val}")
        print()
    
    # ========== CONTROL FLOW ==========
    
    def demonstrate_conditionals(self):
        """Demonstrate conditional statements."""
        print("=== Conditional Statements ===\n")
        
        def classify_number(num):
            """Classify a number as positive, negative, or zero."""
            if num > 0:
                return "positive"
            elif num < 0:
                return "negative"
            else:
                return "zero"
        
        test_numbers = [5, -3, 0, 10, -7]
        for num in test_numbers:
            classification = classify_number(num)
            print(f"{num} is {classification}")
        
        # Nested conditionals
        def grade_student(score):
            """Assign letter grade based on score."""
            if score >= 90:
                if score >= 97:
                    return "A+"
                elif score >= 93:
                    return "A"
                else:
                    return "A-"
            elif score >= 80:
                return "B"
            elif score >= 70:
                return "C"
            elif score >= 60:
                return "D"
            else:
                return "F"
        
        test_scores = [95, 87, 73, 65, 45]
        for score in test_scores:
            grade = grade_student(score)
            print(f"Score {score} → Grade {grade}")
        
        # Ternary operator
        def is_even_ternary(num):
            """Check if number is even using ternary operator."""
            return "even" if num % 2 == 0 else "odd"
        
        print(f"\nTernary operator examples:")
        for num in [2, 3, 4, 5]:
            result = is_even_ternary(num)
            print(f"{num} is {result}")
        print()
    
    def demonstrate_loops(self):
        """Demonstrate different types of loops."""
        print("=== Loops ===\n")
        
        # For loop with range
        print("For loop with range:")
        for i in range(5):
            print(f"  i = {i}")
        
        # For loop with list
        print("\nFor loop with list:")
        fruits = ["apple", "banana", "cherry"]
        for fruit in fruits:
            print(f"  fruit = {fruit}")
        
        # For loop with enumerate
        print("\nFor loop with enumerate:")
        for index, fruit in enumerate(fruits):
            print(f"  {index}: {fruit}")
        
        # While loop
        print("\nWhile loop:")
        count = 0
        while count < 3:
            print(f"  count = {count}")
            count += 1
        
        # Loop with break
        print("\nLoop with break:")
        for i in range(10):
            if i == 3:
                print(f"  Breaking at i = {i}")
                break
            print(f"  i = {i}")
        
        # Loop with continue
        print("\nLoop with continue:")
        for i in range(5):
            if i == 2:
                print(f"  Skipping i = {i}")
                continue
            print(f"  i = {i}")
        
        # Nested loops
        print("\nNested loops:")
        for i in range(3):
            for j in range(2):
                print(f"  i={i}, j={j}")
        print()
    
    # ========== LIST OPERATIONS ==========
    
    def demonstrate_lists(self):
        """Demonstrate list operations."""
        print("=== List Operations ===\n")
        
        # Creating lists
        numbers = [1, 2, 3, 4, 5]
        mixed_list = [1, "hello", 3.14, True]
        empty_list = []
        
        print(f"numbers = {numbers}")
        print(f"mixed_list = {mixed_list}")
        print(f"empty_list = {empty_list}")
        
        # List indexing and slicing
        print(f"\nIndexing and slicing:")
        print(f"numbers[0] = {numbers[0]}")
        print(f"numbers[-1] = {numbers[-1]}")
        print(f"numbers[1:4] = {numbers[1:4]}")
        print(f"numbers[:3] = {numbers[:3]}")
        print(f"numbers[2:] = {numbers[2:]}")
        print(f"numbers[::2] = {numbers[::2]}")
        
        # List methods
        print(f"\nList methods:")
        fruits = ["apple", "banana"]
        print(f"Original: {fruits}")
        
        fruits.append("cherry")
        print(f"After append('cherry'): {fruits}")
        
        fruits.insert(1, "orange")
        print(f"After insert(1, 'orange'): {fruits}")
        
        fruits.remove("banana")
        print(f"After remove('banana'): {fruits}")
        
        popped = fruits.pop()
        print(f"After pop(): {fruits}, popped = {popped}")
        
        # List comprehensions
        print(f"\nList comprehensions:")
        squares = [x**2 for x in range(5)]
        print(f"squares = {squares}")
        
        even_squares = [x**2 for x in range(10) if x % 2 == 0]
        print(f"even_squares = {even_squares}")
        
        # List operations
        print(f"\nList operations:")
        list1 = [1, 2, 3]
        list2 = [4, 5, 6]
        print(f"list1 + list2 = {list1 + list2}")
        print(f"list1 * 3 = {list1 * 3}")
        print(f"len(list1) = {len(list1)}")
        print(f"max(list1) = {max(list1)}")
        print(f"min(list1) = {min(list1)}")
        print(f"sum(list1) = {sum(list1)}")
        print()
    
    def demonstrate_multiple_list_iteration(self):
        """Demonstrate iteration over multiple lists."""
        print("=== Multiple List Iteration ===\n")
        
        # Using zip
        names = ["Alice", "Bob", "Charlie"]
        ages = [25, 30, 35]
        cities = ["New York", "London", "Tokyo"]
        
        print("Using zip with two lists:")
        for name, age in zip(names, ages):
            print(f"  {name} is {age} years old")
        
        print("\nUsing zip with three lists:")
        for name, age, city in zip(names, ages, cities):
            print(f"  {name} is {age} years old and lives in {city}")
        
        # Zip with different length lists
        print("\nZip with different length lists:")
        list1 = [1, 2, 3, 4, 5]
        list2 = ['a', 'b', 'c']
        for num, letter in zip(list1, list2):
            print(f"  {num} -> {letter}")
        
        # Using enumerate with zip
        print("\nUsing enumerate with zip:")
        for i, (name, age) in enumerate(zip(names, ages)):
            print(f"  {i}: {name} is {age} years old")
        
        # Creating dictionary from two lists
        print("\nCreating dictionary from two lists:")
        person_dict = dict(zip(names, ages))
        print(f"  {person_dict}")
        
        # Unzipping lists
        print("\nUnzipping lists:")
        pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
        numbers, letters = zip(*pairs)
        print(f"  pairs = {pairs}")
        print(f"  numbers = {numbers}")
        print(f"  letters = {letters}")
        print()
    
    # ========== STRING OPERATIONS ==========
    
    def demonstrate_strings(self):
        """Demonstrate string operations."""
        print("=== String Operations ===\n")
        
        text = "Hello, World!"
        
        # String methods
        print(f"Original: '{text}'")
        print(f"upper(): '{text.upper()}'")
        print(f"lower(): '{text.lower()}'")
        print(f"title(): '{text.title()}'")
        print(f"replace('World', 'Python'): '{text.replace('World', 'Python')}'")
        
        # String formatting
        name = "Alice"
        age = 30  # Ensure this is an integer
        
        print(f"\nString formatting:")
        print(f"f-string: 'My name is {name} and I am {age} years old'")
        print("format(): 'My name is {} and I am {} years old'".format(name, age))
        print("format() with indices: 'My name is {0} and I am {1} years old'".format(name, age))
        # % formatting (old-style string formatting) - showing the concept
        example_name = "Bob"
        example_age = 25
        percent_formatted = "My name is %s and I am %d years old" % (example_name, example_age)
        print(f"% formatting: {percent_formatted}")
        
        # String operations
        sentence = "The quick brown fox jumps over the lazy dog"
        print(f"\nString operations:")
        print(f"split(): {sentence.split()}")
        print(f"split('o'): {sentence.split('o')}")
        
        words = ["apple", "banana", "cherry"]
        print(f"join(): '{', '.join(words)}'")
        
        # String checking methods
        test_strings = ["hello", "Hello", "123", "hello123", "   ", ""]
        print(f"\nString checking methods:")
        for s in test_strings:
            print(f"  '{s}': islower={s.islower()}, isupper={s.isupper()}, "
                  f"isdigit={s.isdigit()}, isalnum={s.isalnum()}, isspace={s.isspace()}")
        print()


def hackerrank_python_problems():
    """
    Collection of HackerRank-style problems for Python basics.
    """
    
    def problem_1_fizzbuzz():
        """
        Problem 1: FizzBuzz
        
        Print numbers 1 to n, but:
        - Print "Fizz" for multiples of 3
        - Print "Buzz" for multiples of 5
        - Print "FizzBuzz" for multiples of both 3 and 5
        
        Input: n = 15
        """
        def fizzbuzz(n: int) -> List[str]:
            result = []
            for i in range(1, n + 1):
                if i % 15 == 0:
                    result.append("FizzBuzz")
                elif i % 3 == 0:
                    result.append("Fizz")
                elif i % 5 == 0:
                    result.append("Buzz")
                else:
                    result.append(str(i))
            return result
        
        # Test case
        n = 15
        result = fizzbuzz(n)
        print(f"Problem 1 - FizzBuzz(1 to {n}):")
        print(f"  {result}")
        return result
    
    def problem_2_list_operations():
        """
        Problem 2: List Operations
        
        Given a list of integers, perform the following operations:
        1. Remove all even numbers
        2. Square all remaining numbers
        3. Sort in descending order
        
        Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        """
        def process_list(numbers: List[int]) -> List[int]:
            # Remove even numbers
            odd_numbers = [x for x in numbers if x % 2 != 0]
            
            # Square all numbers
            squared = [x**2 for x in odd_numbers]
            
            # Sort in descending order
            squared.sort(reverse=True)
            
            return squared
        
        # Test case
        input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = process_list(input_list)
        print(f"Problem 2 - List Operations:")
        print(f"  Input: {input_list}")
        print(f"  Output: {result}")
        return result
    
    def problem_3_string_manipulation():
        """
        Problem 3: String Manipulation
        
        Given a string, return a new string where:
        1. All vowels are uppercase
        2. All consonants are lowercase
        3. Numbers remain unchanged
        
        Input: "Hello World 123"
        """
        def transform_string(s: str) -> str:
            vowels = "aeiouAEIOU"
            result = []
            
            for char in s:
                if char in vowels:
                    result.append(char.upper())
                elif char.isalpha():
                    result.append(char.lower())
                else:
                    result.append(char)
            
            return ''.join(result)
        
        # Test case
        input_string = "Hello World 123"
        result = transform_string(input_string)
        print(f"Problem 3 - String Manipulation:")
        print(f"  Input: '{input_string}'")
        print(f"  Output: '{result}'")
        return result
    
    def problem_4_nested_loops():
        """
        Problem 4: Pattern Printing
        
        Print a pattern using nested loops:
        *
        **
        ***
        ****
        *****
        
        Input: n = 5
        """
        def print_pattern(n: int) -> List[str]:
            pattern = []
            for i in range(1, n + 1):
                line = '*' * i
                pattern.append(line)
                print(f"  {line}")
            return pattern
        
        # Test case
        n = 5
        print(f"Problem 4 - Pattern Printing (n={n}):")
        result = print_pattern(n)
        return result
    
    def problem_5_function_with_conditions():
        """
        Problem 5: Grade Calculator
        
        Write a function that takes a list of scores and returns:
        - Number of A grades (90-100)
        - Number of B grades (80-89)
        - Number of C grades (70-79)
        - Number of failing grades (<70)
        
        Input: [95, 87, 73, 65, 92, 78, 45, 88]
        """
        def calculate_grades(scores: List[int]) -> dict:
            grade_counts = {'A': 0, 'B': 0, 'C': 0, 'F': 0}
            
            for score in scores:
                if score >= 90:
                    grade_counts['A'] += 1
                elif score >= 80:
                    grade_counts['B'] += 1
                elif score >= 70:
                    grade_counts['C'] += 1
                else:
                    grade_counts['F'] += 1
            
            return grade_counts
        
        # Test case
        scores = [95, 87, 73, 65, 92, 78, 45, 88]
        result = calculate_grades(scores)
        print(f"Problem 5 - Grade Calculator:")
        print(f"  Scores: {scores}")
        print(f"  Grade distribution: {result}")
        return result
    
    def problem_6_zip_and_enumerate():
        """
        Problem 6: Student Data Processing
        
        Given three lists (names, subjects, scores), create a formatted report
        showing each student's performance.
        
        Input: 
        names = ["Alice", "Bob", "Charlie"]
        subjects = ["Math", "Science", "English"]
        scores = [95, 87, 92]
        """
        def create_report(names: List[str], subjects: List[str], scores: List[int]) -> List[str]:
            report = []
            
            for i, (name, subject, score) in enumerate(zip(names, subjects, scores), 1):
                grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'F'
                line = f"{i}. {name} - {subject}: {score} ({grade})"
                report.append(line)
            
            return report
        
        # Test case
        names = ["Alice", "Bob", "Charlie"]
        subjects = ["Math", "Science", "English"]
        scores = [95, 87, 92]
        
        result = create_report(names, subjects, scores)
        print(f"Problem 6 - Student Report:")
        for line in result:
            print(f"  {line}")
        return result
    
    def problem_7_loop_control():
        """
        Problem 7: Prime Number Finder
        
        Find all prime numbers up to n using loop control statements.
        
        Input: n = 20
        """
        def find_primes(n: int) -> List[int]:
            primes = []
            
            for num in range(2, n + 1):
                is_prime = True
                
                # Check if num is divisible by any number from 2 to sqrt(num)
                for i in range(2, int(num**0.5) + 1):
                    if num % i == 0:
                        is_prime = False
                        break  # Exit inner loop early
                
                if is_prime:
                    primes.append(num)
            
            return primes
        
        # Test case
        n = 20
        result = find_primes(n)
        print(f"Problem 7 - Prime Numbers up to {n}:")
        print(f"  {result}")
        return result
    
    # Run all problems
    print("=== HackerRank Style Python Problems ===\n")
    problem_1_fizzbuzz()
    print()
    problem_2_list_operations()
    print()
    problem_3_string_manipulation()
    print()
    problem_4_nested_loops()
    print()
    problem_5_function_with_conditions()
    print()
    problem_6_zip_and_enumerate()
    print()
    problem_7_loop_control()


def advanced_python_concepts():
    """
    Demonstrate some advanced Python concepts useful for interviews.
    """
    print("=== Advanced Python Concepts ===\n")
    
    # Lambda functions
    print("Lambda functions:")
    square = lambda x: x**2
    print(f"square = lambda x: x**2")
    print(f"square(5) = {square(5)}")
    
    numbers = [1, 2, 3, 4, 5]
    squared = list(map(lambda x: x**2, numbers))
    print(f"map(lambda x: x**2, {numbers}) = {squared}")
    
    even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
    print(f"filter(lambda x: x % 2 == 0, {numbers}) = {even_numbers}")
    
    # List comprehensions vs generator expressions
    print(f"\nList comprehensions vs generators:")
    list_comp = [x**2 for x in range(5)]
    gen_exp = (x**2 for x in range(5))
    print(f"List comprehension: {list_comp}")
    print(f"Generator expression: {list(gen_exp)}")
    
    # Dictionary comprehensions
    print(f"\nDictionary comprehensions:")
    word_lengths = {word: len(word) for word in ["apple", "banana", "cherry"]}
    print(f"word_lengths = {word_lengths}")
    
    # Set comprehensions
    print(f"\nSet comprehensions:")
    unique_squares = {x**2 for x in [-2, -1, 0, 1, 2]}
    print(f"unique_squares = {unique_squares}")
    
    # Unpacking
    print(f"\nUnpacking:")
    def get_name_age():
        return "Alice", 25
    
    name, age = get_name_age()
    print(f"name, age = get_name_age() → name='{name}', age={age}")
    
    # Multiple assignment
    a, b, c = 1, 2, 3
    print(f"a, b, c = 1, 2, 3 → a={a}, b={b}, c={c}")
    
    # Swapping variables
    x, y = 10, 20
    print(f"Before swap: x={x}, y={y}")
    x, y = y, x
    print(f"After swap: x={x}, y={y}")
    print()


if __name__ == "__main__":
    # Create demo instance
    demo = PythonBasicsDemo()
    
    # Run all demonstrations
    demo.demonstrate_scalar_types()
    demo.demonstrate_operators()
    demo.demonstrate_functions()
    demo.demonstrate_conditionals()
    demo.demonstrate_loops()
    demo.demonstrate_lists()
    demo.demonstrate_multiple_list_iteration()
    demo.demonstrate_strings()
    
    print("="*60 + "\n")
    
    # Run HackerRank-style problems
    hackerrank_python_problems()
    
    print("\n" + "="*60 + "\n")
    
    # Show advanced concepts
    advanced_python_concepts()
