# Python Technical Interview Preparation Plan
## Data Scientist Position 

---

## 1. Python Fundamentals & Syntax

### 1.1 Core Data Types
- **Primitive Types**: int, float, str, bool, None
- **Type Conversion**: int(), float(), str(), bool()
- **Type Checking**: isinstance(), type()
- **Immutability**: Understanding mutable vs immutable types

### 1.2 Operators & Expressions
- **Arithmetic**: +, -, *, /, //, %, **
- **Comparison**: ==, !=, <, >, <=, >=, is, is not
- **Logical**: and, or, not
- **Membership**: in, not in
- **Bitwise**: &, |, ^, ~, <<, >>
- **Assignment**: =, +=, -=, *=, /=, //=, %=, **=

### 1.3 Control Flow
- **Conditionals**: if, elif, else
- **Loops**: for, while
- **Loop Control**: break, continue, pass
- **Ternary Operator**: x if condition else y
- **Nested Structures**: Nested loops and conditionals

### 1.4 Functions
- **Function Definition**: def, parameters, return
- **Arguments**: Positional, keyword, default parameters
- **Variable Arguments**: *args, **kwargs
- **Lambda Functions**: Anonymous functions, map, filter, reduce
- **Scope**: Local, global, nonlocal, LEGB rule
- **Function Annotations**: Type hints
- **Recursion**: Base cases, recursive calls

---

## 2. Data Structures

### 2.1 Lists
- **Creation**: List literals, list(), list comprehensions
- **Indexing & Slicing**: Positive/negative indices, slice notation
- **Methods**: append, extend, insert, remove, pop, index, count, sort, reverse
- **List Comprehensions**: Basic, nested, conditional
- **Iteration**: for loops, enumerate(), zip()

### 2.2 Tuples
- **Creation**: Tuple literals, tuple()
- **Immutability**: When and why to use tuples
- **Unpacking**: Multiple assignment, tuple unpacking
- **Methods**: count, index

### 2.3 Dictionaries
- **Creation**: Dict literals, dict(), dict comprehensions
- **Access**: dict[key], dict.get(), dict.setdefault()
- **Methods**: keys(), values(), items(), update(), pop(), popitem()
- **Dictionary Comprehensions**: Basic, conditional
- **Nested Dictionaries**: Accessing nested values
- **DefaultDict**: collections.defaultdict

### 2.4 Sets
- **Creation**: Set literals, set(), set comprehensions
- **Operations**: Union, intersection, difference, symmetric_difference
- **Methods**: add, remove, discard, pop, clear, update
- **Set Comprehensions**: Basic usage

### 2.5 Strings
- **String Methods**: upper, lower, strip, split, join, replace, find, index
- **String Formatting**: f-strings, .format(), % formatting
- **String Checking**: isdigit, isalpha, isalnum, isspace, startswith, endswith
- **String Manipulation**: Slicing, concatenation, repetition
- **Regular Expressions**: re module basics (match, search, findall, sub)

### 2.6 Advanced Collections
- **collections.Counter**: Frequency counting
- **collections.defaultdict**: Default values
- **collections.deque**: Double-ended queue
- **collections.namedtuple**: Named tuples

---

## 3. Object-Oriented Programming (OOP)

### 3.1 Classes & Objects
- **Class Definition**: class keyword, __init__
- **Instance Variables**: self, instance attributes
- **Class Variables**: Class attributes vs instance attributes
- **Methods**: Instance methods, class methods (@classmethod), static methods (@staticmethod)
- **Magic Methods**: __str__, __repr__, __len__, __eq__, __lt__, __hash__

### 3.2 Inheritance
- **Single Inheritance**: Base classes, derived classes
- **Multiple Inheritance**: Method Resolution Order (MRO)
- **Method Overriding**: super() function
- **Abstract Base Classes**: abc module basics

### 3.3 Encapsulation
- **Access Modifiers**: Public, protected (_), private (__)
- **Properties**: @property decorator, getters, setters

### 3.4 Polymorphism
- **Duck Typing**: "If it walks like a duck..."
- **Method Overloading**: Not directly supported, alternatives
- **Operator Overloading**: __add__, __sub__, etc.

---

## 4. Advanced Python Features

### 4.1 List Comprehensions & Generator Expressions
- **List Comprehensions**: Basic, nested, conditional, multiple loops
- **Generator Expressions**: Memory-efficient iteration
- **Generators**: yield keyword, generator functions

### 4.2 Decorators
- **Function Decorators**: @decorator syntax
- **Decorator Functions**: Writing custom decorators
- **Decorator Classes**: Class-based decorators
- **Built-in Decorators**: @property, @staticmethod, @classmethod, @functools.lru_cache

### 4.3 Context Managers
- **with Statement**: File handling, resource management
- **Custom Context Managers**: __enter__, __exit__
- **contextlib**: @contextmanager decorator

### 4.4 Iterators & Iterables
- **Iterables**: Objects that can be iterated
- **Iterators**: __iter__, __next__
- **Built-in Functions**: iter(), next(), enumerate(), zip()

### 4.5 Exception Handling
- **try-except**: Basic exception handling
- **Exception Types**: ValueError, TypeError, KeyError, IndexError, etc.
- **Multiple Exceptions**: Handling multiple exception types
- **else & finally**: else clause, finally clause
- **Custom Exceptions**: Creating custom exception classes
- **raise**: Raising exceptions

### 4.6 Modules & Packages
- **Import Statements**: import, from...import, import as
- **Module Search Path**: sys.path
- **__init__.py**: Package initialization
- **Standard Library**: Common modules (os, sys, datetime, json, csv, etc.)

---

## 5. File I/O & Data Handling

### 5.1 File Operations
- **Reading Files**: open(), read(), readline(), readlines()
- **Writing Files**: write(), writelines()
- **File Modes**: 'r', 'w', 'a', 'r+', 'rb', 'wb'
- **Context Managers**: Using with statement for files

### 5.2 CSV Processing
- **csv Module**: csv.reader(), csv.writer(), csv.DictReader(), csv.DictWriter()
- **Reading CSV Files**: Handling headers, data types
- **Writing CSV Files**: Formatting output

### 5.3 JSON Processing
- **json Module**: json.load(), json.dumps(), json.loads(), json.dump()
- **Parsing JSON**: Reading JSON files, parsing JSON strings
- **Creating JSON**: Converting Python objects to JSON

---

## 6. Algorithms & Problem Solving (Easy to Intermediate)

### 6.1 Array/String Manipulation
- **Two Pointers**: Finding pairs, palindromes
- **Sliding Window**: Subarray problems, substring problems
- **Prefix Sum**: Cumulative sums
- **String Parsing**: Extracting information from strings

### 6.2 Searching & Sorting
- **Linear Search**: O(n) search
- **Binary Search**: O(log n) search on sorted arrays
- **Built-in Sorting**: sorted(), list.sort()
- **Custom Sorting**: key parameter, lambda functions

### 6.3 Hash Tables & Sets
- **Frequency Counting**: Using dictionaries/Counter
- **Set Operations**: Finding unique elements, intersections
- **Lookup Optimization**: O(1) lookups

### 6.4 Greedy Algorithms
- **Greedy Approach**: Making locally optimal choices
- **Common Patterns**: Interval scheduling, coin change (simple)

### 6.5 Dynamic Programming (Basic)
- **Memoization**: Caching results
- **Tabulation**: Bottom-up approach
- **Common Patterns**: Fibonacci, coin change, longest common subsequence (basic)

### 6.6 Graph Algorithms (Basic)
- **Graph Representation**: Adjacency list, adjacency matrix
- **BFS**: Breadth-first search
- **DFS**: Depth-first search
- **Tree Traversal**: Inorder, preorder, postorder

---

## 7. Common Problem Patterns

### 7.1 String Problems
- **Anagram Detection**: Checking if strings are anagrams
- **Palindrome Checking**: Reversing strings, checking palindromes
- **String Matching**: Finding substrings, pattern matching
- **String Transformation**: Replacing, splitting, joining

### 7.2 Array/List Problems
- **Finding Elements**: Maximum, minimum, duplicates
- **Array Manipulation**: Rotating, reversing, partitioning
- **Subarray Problems**: Maximum sum, contiguous subarrays
- **Two Sum Variations**: Finding pairs with given sum

### 7.3 Dictionary Problems
- **Frequency Analysis**: Counting occurrences
- **Grouping**: Grouping elements by key
- **Lookup Problems**: Fast lookups, caching

### 7.4 Mathematical Problems
- **Number Operations**: Prime checking, factorial, GCD/LCM
- **Digit Manipulation**: Extracting digits, reversing numbers
- **Mathematical Formulas**: Implementing formulas

---

## 8. Python Best Practices & Code Quality

### 8.1 Code Style
- **PEP 8**: Python style guide basics
- **Naming Conventions**: Variables, functions, classes
- **Line Length**: 79-99 character limit
- **Indentation**: 4 spaces (not tabs)

### 8.2 Code Organization
- **Functions**: Single responsibility, small functions
- **DRY Principle**: Don't Repeat Yourself
- **Comments**: Docstrings, inline comments
- **Modularity**: Breaking code into functions/modules

### 8.3 Performance Considerations
- **Time Complexity**: Big O notation basics
- **Space Complexity**: Memory usage
- **List vs Set**: When to use which
- **Comprehensions vs Loops**: Performance trade-offs

### 8.4 Debugging
- **print() Statements**: Debugging output
- **Common Errors**: SyntaxError, TypeError, ValueError, KeyError, IndexError
- **Error Messages**: Understanding tracebacks

---

## 9. Data Science Libraries (Basics)

### 9.1 NumPy Basics
- **Arrays**: Creating arrays, array operations
- **Array Indexing**: Slicing, boolean indexing
- **Array Methods**: shape, dtype, reshape
- **Basic Operations**: Element-wise operations, broadcasting

### 9.2 Pandas Basics
- **Series**: Creating, indexing, basic operations
- **DataFrame**: Creating, indexing, selecting columns/rows
- **Basic Operations**: head(), tail(), describe(), info()
- **Filtering**: Boolean indexing, query()

### 9.3 Common Operations
- **Data Cleaning**: Handling missing values (basics)
- **Aggregations**: sum(), mean(), count(), groupby() basics
- **Merging**: Basic join operations

---

## 10. Testing Basics

### 10.1 Unit Testing
- **unittest Module**: TestCase, assert methods
- **Writing Tests**: Test functions, test classes
- **Running Tests**: unittest.main(), pytest basics

### 10.2 Test Cases
- **Edge Cases**: Empty inputs, single elements, large inputs
- **Boundary Conditions**: Min/max values
- **Error Cases**: Invalid inputs, exceptions

---

## 11. Common Interview Questions & Patterns

### 11.1 Verbal Questions
- **Python Concepts**: Explain concepts (list vs tuple, mutable vs immutable)
- **Code Review**: Spot bugs, suggest improvements
- **Design Decisions**: Why use certain data structures
- **Best Practices**: Code quality, performance

### 11.2 Coding Patterns
- **String Manipulation**: Parsing, formatting, validation
- **Data Processing**: Filtering, transforming, aggregating
- **Algorithm Implementation**: Sorting, searching, counting
- **Problem Solving**: Breaking down problems, step-by-step approach

### 11.3 Live Coding Tips
- **Communication**: Explain your thought process
- **Start Simple**: Get working solution first, optimize later
- **Test Cases**: Think about edge cases
- **Code Cleanliness**: Write readable code

---

## 12. Practice Resources & Strategy

### 12.1 Practice Platforms
- **HackerRank**: Python problems, scripting challenges
- **LeetCode**: Easy to Medium problems
- **Codewars**: Python katas
- **Practice Problems**: String manipulation, array problems, dictionary problems

### 12.2 Study Approach
- **Daily Practice**: 2-3 problems per day
- **Topic Focus**: One topic per day
- **Mock Interviews**: Practice explaining solutions
- **Time Management**: Practice under time constraints

### 12.3 Review Checklist
- [ ] Python fundamentals (syntax, data types, operators)
- [ ] Data structures (lists, dicts, sets, tuples, strings)
- [ ] Control flow (conditionals, loops)
- [ ] Functions (definition, arguments, scope)
- [ ] OOP basics (classes, inheritance, methods)
- [ ] Advanced features (comprehensions, decorators, generators)
- [ ] Exception handling
- [ ] File I/O (reading/writing files, CSV, JSON)
- [ ] Common algorithms (searching, sorting, basic DP)
- [ ] Problem-solving patterns
- [ ] Code quality and best practices
- [ ] Data science libraries basics (NumPy, Pandas)

---

## 13. Interview Day Preparation

### 13.1 Before Interview
- **Environment Setup**: Python IDE/editor ready
- **Test Environment**: Ensure Python is installed and working
- **Practice Warm-up**: Solve 1-2 easy problems before interview
- **Review Notes**: Quick review of key concepts

### 13.2 During Interview
- **Listen Carefully**: Understand requirements fully
- **Ask Questions**: Clarify requirements if unclear
- **Think Aloud**: Explain your approach
- **Start Coding**: Begin with simple solution
- **Test**: Verify solution with examples
- **Optimize**: Improve if time permits

### 13.3 Common Mistakes to Avoid
- **Not Reading Carefully**: Misunderstanding requirements
- **Jumping to Code**: Not planning first
- **Not Testing**: Forgetting edge cases
- **Poor Communication**: Not explaining thought process
- **Syntax Errors**: Common typos, indentation issues

---

## 14. Topic-by-Topic Study Plan

### Week 1: Fundamentals
- Day 1: Python syntax, data types, operators
- Day 2: Control flow (if/else, loops)
- Day 3: Functions (definition, arguments, scope)
- Day 4: Lists and list comprehensions
- Day 5: Strings and string methods
- Day 6: Dictionaries and sets
- Day 7: Practice problems (fundamentals)

### Week 2: Data Structures & OOP
- Day 1: Advanced list operations, tuples
- Day 2: Dictionary operations, Counter, defaultdict
- Day 3: OOP basics (classes, objects, methods)
- Day 4: Inheritance and polymorphism
- Day 5: Magic methods, properties
- Day 6: Practice problems (data structures)
- Day 7: Practice problems (OOP)

### Week 3: Advanced Features & Algorithms
- Day 1: Generators, iterators, decorators
- Day 2: Exception handling, context managers
- Day 3: File I/O, CSV, JSON processing
- Day 4: Searching algorithms (linear, binary)
- Day 5: Sorting algorithms, built-in sorting
- Day 6: Hash tables, frequency counting
- Day 7: Practice problems (algorithms)

### Week 4: Problem Solving & Practice
- Day 1: String manipulation problems
- Day 2: Array/list problems
- Day 3: Dictionary problems
- Day 4: Mixed problems (easy)
- Day 5: Mixed problems (intermediate)
- Day 6: Mock interview practice
- Day 7: Review and final preparation

---

## 15. Key Concepts Quick Reference

### 15.1 Time Complexities
- **List Operations**: append O(1), insert O(n), search O(n)
- **Dictionary Operations**: get O(1), set O(1), delete O(1)
- **Set Operations**: add O(1), membership test O(1)
- **String Operations**: Most operations O(n)

### 15.2 Common Patterns
- **Frequency Counting**: Use Counter or dict
- **Two Pointers**: Array/string problems
- **Sliding Window**: Subarray/substring problems
- **Hash Map**: Fast lookups, O(1) access

### 15.3 Python Idioms
- **List Comprehension**: [x for x in iterable if condition]
- **Dictionary Comprehension**: {k: v for k, v in items}
- **Enumerate**: for i, item in enumerate(items)
- **Zip**: for a, b in zip(list1, list2)
- **Unpacking**: a, b = tuple

---

**Total Topics: 15 major sections**
**Estimated Preparation Time: 3-4 weeks**
**Focus Areas: Fundamentals, Data Structures, Algorithms (Easy-Intermediate), Problem Solving**
