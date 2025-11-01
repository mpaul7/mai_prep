"""
Python Types: Strings, Collections, and Iteration for Data Science Interview Preparation

This module covers comprehensive Python type system concepts commonly tested in 
data science interviews, particularly HackerRank-style problems.

Topics covered:
1. Scalar Types (int, float, str, bool, None)
2. String Operations and Methods
3. Collections (list, tuple, dict, set)
4. Mutability vs Immutability
5. Iteration and Iterables
6. Collection Methods and Operations
7. Type Conversion and Checking
8. Advanced Collection Techniques
"""

import sys
from typing import List, Dict, Set, Tuple, Union, Any, Optional, Iterable
from collections import Counter, defaultdict, deque, namedtuple
import copy


class PythonTypesDemo:
    """
    A comprehensive demonstration of Python types for interview preparation.
    """
    
    def __init__(self):
        """Initialize the Python types demonstration."""
        pass
    
    # ========== SCALAR TYPES ==========
    
    def demonstrate_scalar_types(self):
        """Demonstrate Python scalar types and their properties."""
        print("=== Python Scalar Types ===\n")
        
        # Integer
        int_val = 42
        print(f"Integer: {int_val}")
        print(f"  Type: {type(int_val)}")
        print(f"  Size in bytes: {sys.getsizeof(int_val)}")
        print(f"  Operations: {int_val + 8}, {int_val * 2}, {int_val ** 2}")
        print(f"  Bit operations: {int_val & 15}, {int_val | 15}, {int_val ^ 15}")
        
        # Float
        float_val = 3.14159
        print(f"\nFloat: {float_val}")
        print(f"  Type: {type(float_val)}")
        import math
        print(f"  Is finite: {math.isfinite(float_val)}")
        print(f"  As integer ratio: {float_val.as_integer_ratio()}")
        print(f"  Hex representation: {float_val.hex()}")
        
        # String
        str_val = "Hello, World!"
        print(f"\nString: '{str_val}'")
        print(f"  Type: {type(str_val)}")
        print(f"  Length: {len(str_val)}")
        print(f"  Encoded: {str_val.encode('utf-8')}")
        print(f"  Is ASCII: {str_val.isascii()}")
        
        # Boolean
        bool_val = True
        print(f"\nBoolean: {bool_val}")
        print(f"  Type: {type(bool_val)}")
        print(f"  Is instance of int: {isinstance(bool_val, int)}")
        print(f"  Integer value: {int(bool_val)}")
        
        # None
        none_val = None
        print(f"\nNone: {none_val}")
        print(f"  Type: {type(none_val)}")
        print(f"  Is None: {none_val is None}")
        print(f"  Boolean value: {bool(none_val)}")
        print()
    
    # ========== STRING OPERATIONS ==========
    
    def demonstrate_strings(self):
        """Demonstrate comprehensive string operations."""
        print("=== String Operations ===\n")
        
        # String creation and basic operations
        text = "Python Programming"
        print(f"Original string: '{text}'")
        print(f"Length: {len(text)}")
        print(f"Character at index 7: '{text[7]}'")
        print(f"Slice [0:6]: '{text[0:6]}'")
        print(f"Slice [7:]: '{text[7:]}'")
        print(f"Slice [::2]: '{text[::2]}'")
        print(f"Reversed: '{text[::-1]}'")
        
        # String methods
        print(f"\nString methods:")
        print(f"  upper(): '{text.upper()}'")
        print(f"  lower(): '{text.lower()}'")
        print(f"  title(): '{text.title()}'")
        print(f"  swapcase(): '{text.swapcase()}'")
        print(f"  capitalize(): '{text.capitalize()}'")
        
        # String checking methods
        test_strings = ["hello", "Hello123", "123", "hello world", "HELLO"]
        print(f"\nString checking methods:")
        for s in test_strings:
            print(f"  '{s}': isalpha={s.isalpha()}, isdigit={s.isdigit()}, "
                  f"isalnum={s.isalnum()}, islower={s.islower()}, isupper={s.isupper()}")
        
        # String searching and manipulation
        sentence = "The quick brown fox jumps over the lazy dog"
        print(f"\nString searching in: '{sentence}'")
        print(f"  find('fox'): {sentence.find('fox')}")
        print(f"  rfind('the'): {sentence.rfind('the')}")
        print(f"  count('the'): {sentence.count('the')}")
        print(f"  startswith('The'): {sentence.startswith('The')}")
        print(f"  endswith('dog'): {sentence.endswith('dog')}")
        
        # String splitting and joining
        words = sentence.split()
        print(f"\nSplit into words: {words[:5]}... (first 5)")
        print(f"Join with '-': '{'-'.join(words[:3])}'")
        
        # String formatting
        name = "Alice"
        age = 30
        score = 95.67
        print(f"\nString formatting:")
        print(f"  f-string: 'Hello {name}, you are {age} years old'")
        print(f"  format(): 'Score: {score:.1f}%'")
        print(f"  zfill(): '{'42'.zfill(5)}'")
        print(f"  center(): '{'Python'.center(15, '*')}'")
        print()
    
    # ========== LISTS ==========
    
    def demonstrate_lists(self):
        """Demonstrate list operations and properties."""
        print("=== Lists (Mutable Sequences) ===\n")
        
        # List creation
        numbers = [1, 2, 3, 4, 5]
        mixed = [1, "hello", 3.14, True, None]
        nested = [[1, 2], [3, 4], [5, 6]]
        
        print(f"Numbers list: {numbers}")
        print(f"Mixed list: {mixed}")
        print(f"Nested list: {nested}")
        print(f"Empty list: {[]}")
        
        # List indexing and slicing
        print(f"\nIndexing and slicing:")
        print(f"  numbers[0]: {numbers[0]}")
        print(f"  numbers[-1]: {numbers[-1]}")
        print(f"  numbers[1:4]: {numbers[1:4]}")
        print(f"  numbers[::2]: {numbers[::2]}")
        
        # List methods (mutating)
        fruits = ["apple", "banana"]
        print(f"\nList methods (original: {fruits}):")
        
        fruits.append("cherry")
        print(f"  After append('cherry'): {fruits}")
        
        fruits.insert(1, "orange")
        print(f"  After insert(1, 'orange'): {fruits}")
        
        fruits.extend(["grape", "kiwi"])
        print(f"  After extend(['grape', 'kiwi']): {fruits}")
        
        removed = fruits.remove("banana")
        print(f"  After remove('banana'): {fruits}")
        
        popped = fruits.pop()
        print(f"  After pop(): {fruits}, popped: {popped}")
        
        popped_index = fruits.pop(1)
        print(f"  After pop(1): {fruits}, popped: {popped_index}")
        
        # List methods (non-mutating)
        numbers = [3, 1, 4, 1, 5, 9, 2, 6]
        print(f"\nList analysis methods (numbers: {numbers}):")
        print(f"  count(1): {numbers.count(1)}")
        print(f"  index(4): {numbers.index(4)}")
        print(f"  min(): {min(numbers)}")
        print(f"  max(): {max(numbers)}")
        print(f"  sum(): {sum(numbers)}")
        
        # List sorting
        numbers_copy = numbers.copy()
        numbers_copy.sort()
        print(f"  sorted (in-place): {numbers_copy}")
        print(f"  sorted (new list): {sorted(numbers, reverse=True)}")
        
        # List comprehensions
        print(f"\nList comprehensions:")
        squares = [x**2 for x in range(5)]
        print(f"  squares: {squares}")
        
        even_squares = [x**2 for x in range(10) if x % 2 == 0]
        print(f"  even squares: {even_squares}")
        
        matrix = [[i*j for j in range(3)] for i in range(3)]
        print(f"  multiplication table: {matrix}")
        print()
    
    # ========== TUPLES ==========
    
    def demonstrate_tuples(self):
        """Demonstrate tuple operations and properties."""
        print("=== Tuples (Immutable Sequences) ===\n")
        
        # Tuple creation
        coordinates = (3, 4)
        colors = ("red", "green", "blue")
        mixed_tuple = (1, "hello", 3.14, True)
        single_element = (42,)  # Note the comma
        empty_tuple = ()
        
        print(f"Coordinates: {coordinates}")
        print(f"Colors: {colors}")
        print(f"Mixed tuple: {mixed_tuple}")
        print(f"Single element: {single_element}")
        print(f"Empty tuple: {empty_tuple}")
        
        # Tuple indexing and slicing
        print(f"\nTuple indexing and slicing:")
        print(f"  colors[0]: {colors[0]}")
        print(f"  colors[-1]: {colors[-1]}")
        print(f"  colors[1:]: {colors[1:]}")
        
        # Tuple methods
        numbers = (1, 2, 3, 2, 4, 2, 5)
        print(f"\nTuple methods (numbers: {numbers}):")
        print(f"  count(2): {numbers.count(2)}")
        print(f"  index(3): {numbers.index(3)}")
        print(f"  len(): {len(numbers)}")
        
        # Tuple unpacking
        point = (10, 20)
        x, y = point
        print(f"\nTuple unpacking:")
        print(f"  point = {point}")
        print(f"  x, y = point → x={x}, y={y}")
        
        # Multiple assignment
        a, b, c = 1, 2, 3
        print(f"  a, b, c = 1, 2, 3 → a={a}, b={b}, c={c}")
        
        # Swapping variables
        x, y = y, x
        print(f"  After swap: x={x}, y={y}")
        
        # Named tuples
        from collections import namedtuple
        Person = namedtuple('Person', ['name', 'age', 'city'])
        person = Person('Alice', 30, 'New York')
        print(f"\nNamed tuple:")
        print(f"  person: {person}")
        print(f"  person.name: {person.name}")
        print(f"  person[1]: {person[1]}")
        print()
    
    # ========== DICTIONARIES ==========
    
    def demonstrate_dictionaries(self):
        """Demonstrate dictionary operations and properties."""
        print("=== Dictionaries (Mutable Mappings) ===\n")
        
        # Dictionary creation
        student = {"name": "Alice", "age": 20, "grade": "A"}
        numbers = {1: "one", 2: "two", 3: "three"}
        mixed_keys = {"str": 1, 42: "int", (1, 2): "tuple"}
        
        print(f"Student dict: {student}")
        print(f"Numbers dict: {numbers}")
        print(f"Mixed keys dict: {mixed_keys}")
        
        # Dictionary access
        print(f"\nDictionary access:")
        print(f"  student['name']: {student['name']}")
        print(f"  student.get('age'): {student.get('age')}")
        print(f"  student.get('height', 'Unknown'): {student.get('height', 'Unknown')}")
        
        # Dictionary methods
        print(f"\nDictionary methods:")
        print(f"  keys(): {list(student.keys())}")
        print(f"  values(): {list(student.values())}")
        print(f"  items(): {list(student.items())}")
        
        # Dictionary modification
        student_copy = student.copy()
        print(f"\nDictionary modification (original: {student_copy}):")
        
        student_copy["height"] = 165
        print(f"  After student['height'] = 165: {student_copy}")
        
        student_copy.update({"weight": 60, "age": 21})
        print(f"  After update(): {student_copy}")
        
        removed = student_copy.pop("weight")
        print(f"  After pop('weight'): {student_copy}, removed: {removed}")
        
        # Dictionary comprehensions
        print(f"\nDictionary comprehensions:")
        squares_dict = {x: x**2 for x in range(5)}
        print(f"  squares: {squares_dict}")
        
        word_lengths = {word: len(word) for word in ["apple", "banana", "cherry"]}
        print(f"  word lengths: {word_lengths}")
        
        # Nested dictionaries
        students = {
            "alice": {"age": 20, "grades": [85, 90, 88]},
            "bob": {"age": 22, "grades": [78, 85, 92]}
        }
        print(f"\nNested dictionary:")
        print(f"  students: {students}")
        print(f"  alice's first grade: {students['alice']['grades'][0]}")
        print()
    
    # ========== SETS ==========
    
    def demonstrate_sets(self):
        """Demonstrate set operations and properties."""
        print("=== Sets (Mutable Collections of Unique Elements) ===\n")
        
        # Set creation
        fruits = {"apple", "banana", "cherry"}
        numbers = {1, 2, 3, 4, 5}
        mixed_set = {1, "hello", 3.14, True}
        empty_set = set()  # Note: {} creates an empty dict, not set
        
        print(f"Fruits set: {fruits}")
        print(f"Numbers set: {numbers}")
        print(f"Mixed set: {mixed_set}")
        print(f"Empty set: {empty_set}")
        
        # Set from list (removes duplicates)
        duplicates = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        unique = set(duplicates)
        print(f"From list with duplicates {duplicates}: {unique}")
        
        # Set methods
        colors = {"red", "green", "blue"}
        print(f"\nSet methods (colors: {colors}):")
        
        colors.add("yellow")
        print(f"  After add('yellow'): {colors}")
        
        colors.update(["orange", "purple"])
        print(f"  After update(['orange', 'purple']): {colors}")
        
        colors.remove("green")  # Raises KeyError if not found
        print(f"  After remove('green'): {colors}")
        
        colors.discard("pink")  # Does not raise error if not found
        print(f"  After discard('pink'): {colors}")
        
        # Set operations
        set1 = {1, 2, 3, 4, 5}
        set2 = {4, 5, 6, 7, 8}
        
        print(f"\nSet operations:")
        print(f"  set1: {set1}")
        print(f"  set2: {set2}")
        print(f"  union (|): {set1 | set2}")
        print(f"  intersection (&): {set1 & set2}")
        print(f"  difference (-): {set1 - set2}")
        print(f"  symmetric difference (^): {set1 ^ set2}")
        
        # Set relationships
        subset = {2, 3}
        print(f"\nSet relationships:")
        print(f"  subset {subset} <= set1 {set1}: {subset <= set1}")
        print(f"  set1 >= subset: {set1 >= subset}")
        print(f"  set1.isdisjoint(set2): {set1.isdisjoint(set2)}")
        
        # Frozen sets (immutable)
        frozen = frozenset([1, 2, 3, 4])
        print(f"\nFrozen set: {frozen}")
        print(f"  Type: {type(frozen)}")
        print()
    
    # ========== MUTABILITY DEMONSTRATION ==========
    
    def demonstrate_mutability(self):
        """Demonstrate mutability vs immutability concepts."""
        print("=== Mutability vs Immutability ===\n")
        
        # Immutable types
        print("Immutable types:")
        
        # Strings
        original_str = "hello"
        modified_str = original_str.upper()
        print(f"  String: original='{original_str}', modified='{modified_str}'")
        print(f"  Same object? {original_str is modified_str}")
        
        # Tuples
        original_tuple = (1, 2, 3)
        # original_tuple[0] = 10  # This would raise TypeError
        print(f"  Tuple: {original_tuple} (cannot be modified in-place)")
        
        # Integers
        x = 10
        y = x
        x += 5
        print(f"  Integer: x={x}, y={y} (y unchanged)")
        
        # Mutable types
        print(f"\nMutable types:")
        
        # Lists
        original_list = [1, 2, 3]
        modified_list = original_list
        modified_list.append(4)
        print(f"  List: original={original_list}, modified={modified_list}")
        print(f"  Same object? {original_list is modified_list}")
        
        # Dictionaries
        original_dict = {"a": 1, "b": 2}
        modified_dict = original_dict
        modified_dict["c"] = 3
        print(f"  Dict: original={original_dict}, modified={modified_dict}")
        print(f"  Same object? {original_dict is modified_dict}")
        
        # Shallow vs Deep copy
        print(f"\nShallow vs Deep copy:")
        
        nested_list = [[1, 2], [3, 4]]
        shallow_copy = nested_list.copy()
        deep_copy = copy.deepcopy(nested_list)
        
        nested_list[0].append(3)
        print(f"  Original: {nested_list}")
        print(f"  Shallow copy: {shallow_copy}")
        print(f"  Deep copy: {deep_copy}")
        
        # Mutable default arguments (common pitfall)
        print(f"\nMutable default arguments pitfall:")
        
        def bad_function(items=[]):
            items.append("new")
            return items
        
        def good_function(items=None):
            if items is None:
                items = []
            items.append("new")
            return items
        
        print(f"  bad_function(): {bad_function()}")
        print(f"  bad_function(): {bad_function()}")  # Accumulates!
        print(f"  good_function(): {good_function()}")
        print(f"  good_function(): {good_function()}")  # Fresh each time
        print()
    
    # ========== ITERATION AND ITERABLES ==========
    
    def demonstrate_iteration(self):
        """Demonstrate iteration concepts and iterables."""
        print("=== Iteration and Iterables ===\n")
        
        # Basic iteration
        print("Basic iteration:")
        numbers = [1, 2, 3, 4, 5]
        
        print(f"  List: {numbers}")
        for num in numbers:
            print(f"    {num}", end=" ")
        print()
        
        # String iteration
        word = "Python"
        print(f"  String: '{word}'")
        for char in word:
            print(f"    '{char}'", end=" ")
        print()
        
        # Dictionary iteration
        student = {"name": "Alice", "age": 20, "grade": "A"}
        print(f"  Dictionary: {student}")
        
        print("    Keys:", end=" ")
        for key in student:
            print(key, end=" ")
        print()
        
        print("    Values:", end=" ")
        for value in student.values():
            print(value, end=" ")
        print()
        
        print("    Items:", end=" ")
        for key, value in student.items():
            print(f"({key}:{value})", end=" ")
        print()
        
        # Enumerate
        print(f"\nEnumerate:")
        colors = ["red", "green", "blue"]
        for i, color in enumerate(colors):
            print(f"    {i}: {color}")
        
        for i, color in enumerate(colors, start=1):
            print(f"    {i}: {color}")
        
        # Zip
        print(f"\nZip:")
        names = ["Alice", "Bob", "Charlie"]
        ages = [25, 30, 35]
        cities = ["NYC", "LA", "Chicago"]
        
        for name, age in zip(names, ages):
            print(f"    {name} is {age} years old")
        
        for name, age, city in zip(names, ages, cities):
            print(f"    {name}, {age}, lives in {city}")
        
        # Range
        print(f"\nRange:")
        print(f"  range(5): {list(range(5))}")
        print(f"  range(2, 8): {list(range(2, 8))}")
        print(f"  range(0, 10, 2): {list(range(0, 10, 2))}")
        
        # Iterators
        print(f"\nIterators:")
        numbers_iter = iter([1, 2, 3])
        print(f"  next(): {next(numbers_iter)}")
        print(f"  next(): {next(numbers_iter)}")
        print(f"  next(): {next(numbers_iter)}")
        # print(f"  next(): {next(numbers_iter)}")  # Would raise StopIteration
        
        # Generator expressions
        print(f"\nGenerator expressions:")
        squares_gen = (x**2 for x in range(5))
        print(f"  Generator: {squares_gen}")
        print(f"  List from generator: {list(squares_gen)}")
        
        # List vs generator memory usage
        import sys
        list_comp = [x**2 for x in range(1000)]
        gen_exp = (x**2 for x in range(1000))
        print(f"  List comprehension size: {sys.getsizeof(list_comp)} bytes")
        print(f"  Generator expression size: {sys.getsizeof(gen_exp)} bytes")
        print()
    
    # ========== ADVANCED COLLECTIONS ==========
    
    def demonstrate_advanced_collections(self):
        """Demonstrate advanced collection types."""
        print("=== Advanced Collections ===\n")
        
        # Counter
        from collections import Counter
        text = "hello world"
        char_count = Counter(text)
        print(f"Counter:")
        print(f"  Text: '{text}'")
        print(f"  Character count: {char_count}")
        print(f"  Most common: {char_count.most_common(3)}")
        
        # DefaultDict
        from collections import defaultdict
        dd = defaultdict(list)
        words = ["apple", "banana", "apricot", "blueberry", "cherry"]
        
        for word in words:
            dd[word[0]].append(word)
        
        print(f"\nDefaultDict:")
        print(f"  Words grouped by first letter: {dict(dd)}")
        
        # Deque (double-ended queue)
        from collections import deque
        dq = deque([1, 2, 3])
        print(f"\nDeque:")
        print(f"  Initial: {dq}")
        
        dq.appendleft(0)
        print(f"  After appendleft(0): {dq}")
        
        dq.append(4)
        print(f"  After append(4): {dq}")
        
        left = dq.popleft()
        right = dq.pop()
        print(f"  After popleft() and pop(): {dq}, removed: {left}, {right}")
        
        # OrderedDict (less relevant in Python 3.7+)
        from collections import OrderedDict
        od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
        print(f"\nOrderedDict: {od}")
        
        # ChainMap
        from collections import ChainMap
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'b': 3, 'c': 4}
        chain = ChainMap(dict1, dict2)
        print(f"\nChainMap:")
        print(f"  dict1: {dict1}")
        print(f"  dict2: {dict2}")
        print(f"  ChainMap: {dict(chain)}")
        print(f"  chain['b']: {chain['b']}")  # First dict takes precedence
        print()


def hackerrank_types_problems():
    """
    Collection of HackerRank-style problems for Python types.
    """
    
    def problem_1_string_operations():
        """
        Problem 1: String Manipulation
        
        Given a string, perform the following operations:
        1. Count vowels and consonants
        2. Reverse words but keep their positions
        3. Find the longest word
        
        Input: "The quick brown fox"
        """
        def analyze_string(s: str) -> Dict[str, Union[int, str]]:
            vowels = "aeiouAEIOU"
            vowel_count = sum(1 for char in s if char in vowels)
            consonant_count = sum(1 for char in s if char.isalpha() and char not in vowels)
            
            words = s.split()
            reversed_words = [word[::-1] for word in words]
            reversed_sentence = " ".join(reversed_words)
            
            longest_word = max(words, key=len) if words else ""
            
            return {
                "vowels": vowel_count,
                "consonants": consonant_count,
                "reversed": reversed_sentence,
                "longest_word": longest_word
            }
        
        # Test case
        test_string = "The quick brown fox"
        result = analyze_string(test_string)
        
        print(f"Problem 1 - String Analysis:")
        print(f"  Input: '{test_string}'")
        print(f"  Vowels: {result['vowels']}")
        print(f"  Consonants: {result['consonants']}")
        print(f"  Reversed words: '{result['reversed']}'")
        print(f"  Longest word: '{result['longest_word']}'")
        return result
    
    def problem_2_list_operations():
        """
        Problem 2: List Manipulation
        
        Given a list of integers:
        1. Remove duplicates while preserving order
        2. Find pairs that sum to a target
        3. Rotate the list by k positions
        
        Input: [1, 2, 3, 2, 4, 3, 5], target=5, k=2
        """
        def manipulate_list(nums: List[int], target: int, k: int) -> Dict[str, Any]:
            # Remove duplicates preserving order
            seen = set()
            unique = []
            for num in nums:
                if num not in seen:
                    seen.add(num)
                    unique.append(num)
            
            # Find pairs that sum to target
            pairs = []
            num_indices = {}
            for i, num in enumerate(unique):
                complement = target - num
                if complement in num_indices:
                    pairs.append((complement, num))
                num_indices[num] = i
            
            # Rotate list by k positions
            if unique:
                k = k % len(unique)
                rotated = unique[-k:] + unique[:-k]
            else:
                rotated = []
            
            return {
                "original": nums,
                "unique": unique,
                "pairs": pairs,
                "rotated": rotated
            }
        
        # Test case
        nums = [1, 2, 3, 2, 4, 3, 5]
        target = 5
        k = 2
        result = manipulate_list(nums, target, k)
        
        print(f"Problem 2 - List Manipulation:")
        print(f"  Original: {result['original']}")
        print(f"  Unique: {result['unique']}")
        print(f"  Pairs summing to {target}: {result['pairs']}")
        print(f"  Rotated by {k}: {result['rotated']}")
        return result
    
    def problem_3_dictionary_operations():
        """
        Problem 3: Dictionary Processing
        
        Given a list of dictionaries representing students:
        1. Group students by grade
        2. Find student with highest average score
        3. Count students per city
        
        Input: [{"name": "Alice", "grade": "A", "scores": [85, 90, 88], "city": "NYC"}, ...]
        """
        def process_students(students: List[Dict]) -> Dict[str, Any]:
            # Group by grade
            grade_groups = {}
            for student in students:
                grade = student.get("grade", "Unknown")
                if grade not in grade_groups:
                    grade_groups[grade] = []
                grade_groups[grade].append(student["name"])
            
            # Find student with highest average
            best_student = None
            highest_avg = -1
            
            for student in students:
                if "scores" in student:
                    avg = sum(student["scores"]) / len(student["scores"])
                    if avg > highest_avg:
                        highest_avg = avg
                        best_student = student["name"]
            
            # Count students per city
            city_counts = {}
            for student in students:
                city = student.get("city", "Unknown")
                city_counts[city] = city_counts.get(city, 0) + 1
            
            return {
                "grade_groups": grade_groups,
                "best_student": best_student,
                "best_average": round(highest_avg, 2),
                "city_counts": city_counts
            }
        
        # Test case
        students = [
            {"name": "Alice", "grade": "A", "scores": [85, 90, 88], "city": "NYC"},
            {"name": "Bob", "grade": "B", "scores": [78, 82, 80], "city": "LA"},
            {"name": "Charlie", "grade": "A", "scores": [92, 95, 89], "city": "NYC"},
            {"name": "Diana", "grade": "B", "scores": [88, 85, 90], "city": "Chicago"}
        ]
        
        result = process_students(students)
        
        print(f"Problem 3 - Dictionary Processing:")
        print(f"  Grade groups: {result['grade_groups']}")
        print(f"  Best student: {result['best_student']} (avg: {result['best_average']})")
        print(f"  City counts: {result['city_counts']}")
        return result
    
    def problem_4_set_operations():
        """
        Problem 4: Set Operations
        
        Given multiple sets of data:
        1. Find common elements across all sets
        2. Find elements unique to each set
        3. Create a power set (all possible subsets)
        
        Input: {1,2,3,4}, {3,4,5,6}, {4,5,6,7}
        """
        def analyze_sets(set1: Set[int], set2: Set[int], set3: Set[int]) -> Dict[str, Any]:
            # Common elements across all sets
            common = set1 & set2 & set3
            
            # Elements unique to each set
            unique_to_1 = set1 - set2 - set3
            unique_to_2 = set2 - set1 - set3
            unique_to_3 = set3 - set1 - set2
            
            # Power set of the first set (limited to small sets)
            def power_set(s):
                if len(s) > 5:  # Limit for demonstration
                    return "Too large to display"
                
                s_list = list(s)
                power_set_list = []
                
                for i in range(2**len(s_list)):
                    subset = []
                    for j in range(len(s_list)):
                        if i & (1 << j):
                            subset.append(s_list[j])
                    power_set_list.append(set(subset))
                
                return power_set_list
            
            return {
                "set1": set1,
                "set2": set2,
                "set3": set3,
                "common": common,
                "unique_to_1": unique_to_1,
                "unique_to_2": unique_to_2,
                "unique_to_3": unique_to_3,
                "power_set_of_1": power_set(set1)
            }
        
        # Test case
        set1 = {1, 2, 3, 4}
        set2 = {3, 4, 5, 6}
        set3 = {4, 5, 6, 7}
        
        result = analyze_sets(set1, set2, set3)
        
        print(f"Problem 4 - Set Operations:")
        print(f"  Set 1: {result['set1']}")
        print(f"  Set 2: {result['set2']}")
        print(f"  Set 3: {result['set3']}")
        print(f"  Common to all: {result['common']}")
        print(f"  Unique to set 1: {result['unique_to_1']}")
        print(f"  Unique to set 2: {result['unique_to_2']}")
        print(f"  Unique to set 3: {result['unique_to_3']}")
        print(f"  Power set of set 1: {result['power_set_of_1']}")
        return result
    
    def problem_5_tuple_operations():
        """
        Problem 5: Tuple Processing
        
        Given a list of tuples representing coordinates:
        1. Find the point closest to origin
        2. Sort points by distance from a given point
        3. Group points by quadrant
        
        Input: [(1,2), (3,4), (-1,2), (0,-3), (2,-1)]
        """
        def process_coordinates(points: List[Tuple[int, int]], 
                              reference: Tuple[int, int] = (0, 0)) -> Dict[str, Any]:
            import math
            
            # Distance from origin
            def distance_from_origin(point):
                return math.sqrt(point[0]**2 + point[1]**2)
            
            # Distance from reference point
            def distance_from_reference(point):
                return math.sqrt((point[0] - reference[0])**2 + (point[1] - reference[1])**2)
            
            # Find closest to origin
            closest_to_origin = min(points, key=distance_from_origin)
            
            # Sort by distance from reference
            sorted_by_distance = sorted(points, key=distance_from_reference)
            
            # Group by quadrant
            quadrants = {"I": [], "II": [], "III": [], "IV": [], "Axes": []}
            
            for point in points:
                x, y = point
                if x > 0 and y > 0:
                    quadrants["I"].append(point)
                elif x < 0 and y > 0:
                    quadrants["II"].append(point)
                elif x < 0 and y < 0:
                    quadrants["III"].append(point)
                elif x > 0 and y < 0:
                    quadrants["IV"].append(point)
                else:
                    quadrants["Axes"].append(point)
            
            return {
                "points": points,
                "closest_to_origin": closest_to_origin,
                "sorted_by_distance": sorted_by_distance,
                "quadrants": {k: v for k, v in quadrants.items() if v},
                "reference_point": reference
            }
        
        # Test case
        points = [(1, 2), (3, 4), (-1, 2), (0, -3), (2, -1)]
        reference = (1, 1)
        
        result = process_coordinates(points, reference)
        
        print(f"Problem 5 - Tuple Processing:")
        print(f"  Points: {result['points']}")
        print(f"  Closest to origin: {result['closest_to_origin']}")
        print(f"  Sorted by distance from {result['reference_point']}: {result['sorted_by_distance']}")
        print(f"  Quadrants: {result['quadrants']}")
        return result
    
    # Run all problems
    print("=== HackerRank Style Python Types Problems ===\n")
    problem_1_string_operations()
    print()
    problem_2_list_operations()
    print()
    problem_3_dictionary_operations()
    print()
    problem_4_set_operations()
    print()
    problem_5_tuple_operations()


if __name__ == "__main__":
    # Create demo instance
    demo = PythonTypesDemo()
    
    # Run all demonstrations
    demo.demonstrate_scalar_types()
    demo.demonstrate_strings()
    demo.demonstrate_lists()
    demo.demonstrate_tuples()
    demo.demonstrate_dictionaries()
    demo.demonstrate_sets()
    demo.demonstrate_mutability()
    demo.demonstrate_iteration()
    demo.demonstrate_advanced_collections()
    
    print("="*60 + "\n")
    
    # Run HackerRank-style problems
    hackerrank_types_problems()
