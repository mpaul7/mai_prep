"""
Python Modules & Packages - Interview Preparation
Topic 4.6: Modules & Packages

This module covers:
- Import Statements: import, from...import, import as
- Module Search Path: sys.path
- __init__.py: Package initialization
- Standard Library: Common modules (os, sys, datetime, json, csv, etc.)
"""

import sys
import os
import datetime
import json
import csv
import math
import random
import time
from pathlib import Path

# ============================================================================
# 1. UNDERSTANDING MODULES AND PACKAGES
# ============================================================================

print("=" * 70)
print("1. UNDERSTANDING MODULES AND PACKAGES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 What is a Module?
# ----------------------------------------------------------------------------
print("\n--- What is a Module? ---")

print("""
A MODULE is a file containing Python definitions and statements.
- File with .py extension
- Can contain functions, classes, variables
- Can be imported and used in other files
- Example: math.py, os.py, sys.py
""")


# ----------------------------------------------------------------------------
# 1.2 What is a Package?
# ----------------------------------------------------------------------------
print("\n--- What is a Package? ---")

print("""
A PACKAGE is a collection of modules organized in directories.
- Directory containing __init__.py file
- Can contain subpackages (nested directories)
- Provides namespace for modules
- Example: collections, os.path, json
""")


# ============================================================================
# 2. IMPORT STATEMENTS
# ============================================================================

print("\n" + "=" * 70)
print("2. IMPORT STATEMENTS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic import
# ----------------------------------------------------------------------------
print("\n--- Basic import ---")

# Import entire module
import math
print(f"math.pi = {math.pi}")
print(f"math.sqrt(16) = {math.sqrt(16)}")

# Import multiple modules
import os
import sys
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")


# ----------------------------------------------------------------------------
# 2.2 from...import
# ----------------------------------------------------------------------------
print("\n--- from...import ---")

# Import specific items
from math import pi, sqrt, sin
print(f"pi = {pi}")  # No need for math. prefix
print(f"sqrt(16) = {sqrt(16)}")
print(f"sin(pi/2) = {sin(pi/2)}")

# Import multiple items
from datetime import date, datetime, timedelta
today = date.today()
print(f"Today: {today}")

# Import all (not recommended - pollutes namespace)
# from math import *


# ----------------------------------------------------------------------------
# 2.3 import as (Aliasing)
# ----------------------------------------------------------------------------
print("\n--- import as (Aliasing) ---")

# Import with alias
import datetime as dt
print(f"Current time: {dt.datetime.now()}")

# Useful for long module names or avoiding conflicts
import pandas as pd  # Common convention
import numpy as np   # Common convention

# Import specific item with alias
from datetime import datetime as dt_class
now = dt_class.now()
print(f"Now: {now}")


# ----------------------------------------------------------------------------
# 2.4 Import Order (PEP 8)
# ----------------------------------------------------------------------------
print("\n--- Import Order (PEP 8) ---")

print("""
PEP 8 import order:
1. Standard library imports
2. Related third party imports
3. Local application/library imports

Example:
import os
import sys

import numpy as np
import pandas as pd

from mypackage import mymodule
""")


# ============================================================================
# 3. MODULE SEARCH PATH (sys.path)
# ============================================================================

print("\n" + "=" * 70)
print("3. MODULE SEARCH PATH (sys.path)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Understanding sys.path
# ----------------------------------------------------------------------------
print("\n--- Understanding sys.path ---")

# sys.path is a list of directories Python searches for modules
print("Python searches for modules in these directories (in order):")
for i, path in enumerate(sys.path, 1):
    print(f"{i}. {path}")

# Current directory is usually first (or empty string)
print(f"\nCurrent working directory: {os.getcwd()}")


# ----------------------------------------------------------------------------
# 3.2 Modifying sys.path
# ----------------------------------------------------------------------------
print("\n--- Modifying sys.path ---")

# Add directory to search path
custom_path = "/tmp/my_modules"
if os.path.exists(custom_path):
    sys.path.insert(0, custom_path)
    print(f"Added {custom_path} to sys.path")

# Using PYTHONPATH environment variable
# export PYTHONPATH=/path/to/modules:$PYTHONPATH


# ----------------------------------------------------------------------------
# 3.3 Finding Module Location
# ----------------------------------------------------------------------------
print("\n--- Finding Module Location ---")

# Get module file location
print(f"math module location: {math.__file__}")
print(f"os module location: {os.__file__}")
print(f"sys module location: {sys.__file__}")


# ----------------------------------------------------------------------------
# 3.4 Module vs Script
# ----------------------------------------------------------------------------
print("\n--- Module vs Script ---")

print("""
Module: File meant to be imported
Script: File meant to be executed directly

Check if running as script:
if __name__ == "__main__":
    # Code that runs only when executed directly
    pass
""")


# ============================================================================
# 4. PACKAGES AND __init__.py
# ============================================================================

print("\n" + "=" * 70)
print("4. PACKAGES AND __init__.py")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 What is __init__.py?
# ----------------------------------------------------------------------------
print("\n--- What is __init__.py? ---")

print("""
__init__.py:
- Makes a directory a Python package
- Executed when package is imported
- Can be empty or contain initialization code
- Can control what gets imported with 'from package import *'
- Python 3.3+: Can use namespace packages (no __init__.py needed)
""")


# ----------------------------------------------------------------------------
# 4.2 Package Structure Example
# ----------------------------------------------------------------------------
print("\n--- Package Structure Example ---")

print("""
Example package structure:

mypackage/
    __init__.py          # Package initialization
    module1.py           # Module 1
    module2.py           # Module 2
    subpackage/
        __init__.py      # Subpackage initialization
        module3.py       # Module 3

Import examples:
import mypackage
from mypackage import module1
from mypackage.subpackage import module3
""")


# ----------------------------------------------------------------------------
# 4.3 __init__.py Content
# ----------------------------------------------------------------------------
print("\n--- __init__.py Content ---")

print("""
__init__.py can:
- Import submodules for convenience
- Define __all__ to control 'from package import *'
- Initialize package-level variables
- Set up package configuration

Example __init__.py:
# mypackage/__init__.py
from .module1 import function1
from .module2 import Class1

__all__ = ['function1', 'Class1']
""")


# ============================================================================
# 5. STANDARD LIBRARY MODULES
# ============================================================================

print("\n" + "=" * 70)
print("5. STANDARD LIBRARY MODULES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 os - Operating System Interface
# ----------------------------------------------------------------------------
print("\n--- os - Operating System Interface ---")

# Current directory
print(f"Current directory: {os.getcwd()}")

# List directory contents
print(f"Files in current directory: {os.listdir('.')[:5]}")  # First 5

# Environment variables
print(f"HOME: {os.environ.get('HOME', 'Not set')}")

# Path operations
file_path = os.path.join("dir", "subdir", "file.txt")
print(f"Joined path: {file_path}")

# Check if path exists
print(f"Path exists: {os.path.exists('.')}")

# File operations
print(f"Is file: {os.path.isfile(__file__)}")
print(f"Is directory: {os.path.isdir('.')}")


# ----------------------------------------------------------------------------
# 5.2 sys - System-Specific Parameters
# ----------------------------------------------------------------------------
print("\n--- sys - System-Specific Parameters ---")

# Python version
print(f"Python version: {sys.version}")

# Command line arguments
print(f"Script name: {sys.argv[0] if sys.argv else 'N/A'}")

# Exit program
# sys.exit(0)  # Exit with code 0

# Standard streams
print(f"stdin: {sys.stdin}")
print(f"stdout: {sys.stdout}")
print(f"stderr: {sys.stderr}")


# ----------------------------------------------------------------------------
# 5.3 datetime - Date and Time
# ----------------------------------------------------------------------------
print("\n--- datetime - Date and Time ---")

# Current date and time
now = datetime.datetime.now()
print(f"Current datetime: {now}")

# Current date
today = datetime.date.today()
print(f"Today: {today}")

# Create specific date
birthday = datetime.date(1990, 5, 15)
print(f"Birthday: {birthday}")

# Time delta
age = today - birthday
print(f"Age in days: {age.days}")

# Formatting
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"Formatted: {formatted}")

# Parsing
parsed = datetime.datetime.strptime("2024-01-15", "%Y-%m-%d")
print(f"Parsed: {parsed}")


# ----------------------------------------------------------------------------
# 5.4 json - JSON Data
# ----------------------------------------------------------------------------
print("\n--- json - JSON Data ---")

# Python dict to JSON string
data = {"name": "Alice", "age": 25, "city": "New York"}
json_string = json.dumps(data)
print(f"JSON string: {json_string}")

# JSON string to Python dict
parsed_data = json.loads(json_string)
print(f"Parsed data: {parsed_data}")

# Pretty printing
pretty_json = json.dumps(data, indent=2)
print(f"Pretty JSON:\n{pretty_json}")

# Writing to file
with open("example.json", "w") as f:
    json.dump(data, f)

# Reading from file
with open("example.json", "r") as f:
    loaded_data = json.load(f)
    print(f"Loaded from file: {loaded_data}")


# ----------------------------------------------------------------------------
# 5.5 csv - CSV File Handling
# ----------------------------------------------------------------------------
print("\n--- csv - CSV File Handling ---")

# Writing CSV
data = [
    ["Name", "Age", "City"],
    ["Alice", "25", "New York"],
    ["Bob", "30", "Los Angeles"],
    ["Charlie", "35", "Chicago"]
]

with open("example.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("CSV file written")

# Reading CSV
with open("example.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(f"Row: {row}")

# DictReader (more convenient)
with open("example.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Dict row: {row}")


# ----------------------------------------------------------------------------
# 5.6 math - Mathematical Functions
# ----------------------------------------------------------------------------
print("\n--- math - Mathematical Functions ---")

print(f"pi = {math.pi}")
print(f"e = {math.e}")
print(f"sqrt(16) = {math.sqrt(16)}")
print(f"log(10) = {math.log(10)}")
print(f"sin(pi/2) = {math.sin(math.pi/2)}")
print(f"ceil(4.3) = {math.ceil(4.3)}")
print(f"floor(4.7) = {math.floor(4.7)}")
print(f"factorial(5) = {math.factorial(5)}")


# ----------------------------------------------------------------------------
# 5.7 random - Random Number Generation
# ----------------------------------------------------------------------------
print("\n--- random - Random Number Generation ---")

# Random float between 0 and 1
print(f"Random float: {random.random()}")

# Random integer in range
print(f"Random int (1-10): {random.randint(1, 10)}")

# Random choice from sequence
choices = ["apple", "banana", "cherry"]
print(f"Random choice: {random.choice(choices)}")

# Shuffle list
numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)
print(f"Shuffled: {numbers}")

# Random sample
sample = random.sample(range(100), 5)
print(f"Random sample: {sample}")


# ----------------------------------------------------------------------------
# 5.8 time - Time-Related Functions
# ----------------------------------------------------------------------------
print("\n--- time - Time-Related Functions ---")

# Current time (seconds since epoch)
current_time = time.time()
print(f"Current time (seconds): {current_time}")

# Sleep
# time.sleep(1)  # Sleep for 1 second

# Formatted time
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f"Formatted time: {formatted_time}")


# ----------------------------------------------------------------------------
# 5.9 collections - Specialized Container Types
# ----------------------------------------------------------------------------
print("\n--- collections - Specialized Container Types ---")

from collections import Counter, defaultdict, deque, namedtuple

# Counter
counter = Counter([1, 2, 2, 3, 3, 3])
print(f"Counter: {counter}")

# defaultdict
dd = defaultdict(int)
dd["a"] += 1
print(f"defaultdict: {dd}")

# deque
dq = deque([1, 2, 3])
dq.appendleft(0)
print(f"deque: {dq}")

# namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(1, 2)
print(f"Point: {p}, x={p.x}, y={p.y}")


# ----------------------------------------------------------------------------
# 5.10 itertools - Iterator Functions
# ----------------------------------------------------------------------------
print("\n--- itertools - Iterator Functions ---")

from itertools import chain, combinations, permutations, cycle

# Chain iterables
chained = list(chain([1, 2], [3, 4]))
print(f"Chained: {chained}")

# Combinations
combs = list(combinations([1, 2, 3], 2))
print(f"Combinations: {combs}")

# Permutations
perms = list(permutations([1, 2], 2))
print(f"Permutations: {perms}")


# ----------------------------------------------------------------------------
# 5.11 re - Regular Expressions
# ----------------------------------------------------------------------------
print("\n--- re - Regular Expressions ---")

import re

# Search pattern
text = "The price is $100"
match = re.search(r'\$(\d+)', text)
if match:
    print(f"Found: {match.group(1)}")

# Find all matches
numbers = re.findall(r'\d+', "I have 5 apples and 3 oranges")
print(f"Numbers: {numbers}")

# Replace
new_text = re.sub(r'\d+', 'X', "I have 5 apples")
print(f"Replaced: {new_text}")


# ----------------------------------------------------------------------------
# 5.12 pathlib - Object-Oriented Filesystem Paths
# ----------------------------------------------------------------------------
print("\n--- pathlib - Object-Oriented Filesystem Paths ---")

# Create path object
path = Path("example.txt")
print(f"Path: {path}")
print(f"Exists: {path.exists()}")
print(f"Is file: {path.is_file()}")

# Path operations
base = Path("/home/user")
full_path = base / "documents" / "file.txt"
print(f"Full path: {full_path}")


# ============================================================================
# 6. ADVANCED IMPORT PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("6. ADVANCED IMPORT PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 6.1 Conditional Imports
# ----------------------------------------------------------------------------
print("\n--- Conditional Imports ---")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Pandas not available")


# ----------------------------------------------------------------------------
# 6.2 Lazy Imports
# ----------------------------------------------------------------------------
print("\n--- Lazy Imports ---")

def lazy_import():
    """Import module only when needed."""
    import json
    return json

# Module imported only when function is called
json_module = lazy_import()


# ----------------------------------------------------------------------------
# 6.3 Importing from Parent Directory
# ----------------------------------------------------------------------------
print("\n--- Importing from Parent Directory ---")

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# ----------------------------------------------------------------------------
# 6.4 __all__ for Package Exports
# ----------------------------------------------------------------------------
print("\n--- __all__ for Package Exports ---")

print("""
__all__ defines what gets imported with 'from package import *'

Example in module:
__all__ = ['function1', 'Class1', 'CONSTANT']

Only items in __all__ are imported with *
""")


# ============================================================================
# 7. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Module Information
print("\n--- Exercise 1: Module Information ---")

def get_module_info(module_name):
    """Get information about a module."""
    try:
        module = __import__(module_name)
        return {
            'name': module_name,
            'file': getattr(module, '__file__', 'Built-in'),
            'doc': module.__doc__ or 'No documentation'
        }
    except ImportError:
        return None

info = get_module_info('math')
if info:
    print(f"Module info: {info}")


# Exercise 2: Safe Import
print("\n--- Exercise 2: Safe Import ---")

def safe_import(module_name, default=None):
    """Safely import a module."""
    try:
        return __import__(module_name)
    except ImportError:
        return default

numpy_module = safe_import('numpy')
if numpy_module:
    print("NumPy imported successfully")
else:
    print("NumPy not available")


# Exercise 3: List Available Modules
print("\n--- Exercise 3: List Available Modules ---")

# Get all standard library modules
import pkgutil
standard_modules = [name for _, name, _ in pkgutil.iter_modules()]
print(f"Number of standard modules: {len(standard_modules)}")
print(f"Sample modules: {standard_modules[:10]}")


# Exercise 4: Dynamic Import
print("\n--- Exercise 4: Dynamic Import ---")

def import_function(module_name, function_name):
    """Dynamically import a function."""
    try:
        module = __import__(module_name, fromlist=[function_name])
        return getattr(module, function_name)
    except (ImportError, AttributeError):
        return None

sqrt_func = import_function('math', 'sqrt')
if sqrt_func:
    print(f"sqrt(16) = {sqrt_func(16)}")


# ============================================================================
# 8. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("8. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between import and from...import?
print("\n--- Q1: import vs from...import ---")
print("""
import module:
- Imports entire module
- Access with module.function()
- Namespace preserved

from module import function:
- Imports specific items
- Access directly with function()
- Can pollute namespace
""")


# Q2: What is __init__.py?
print("\n--- Q2: What is __init__.py? ---")
print("""
__init__.py:
- Makes directory a Python package
- Executed when package is imported
- Can initialize package
- Can define __all__ for imports
- Python 3.3+: Optional (namespace packages)
""")


# Q3: How does Python find modules?
print("\n--- Q3: How does Python find modules? ---")
print("""
Python searches in sys.path (in order):
1. Current directory
2. PYTHONPATH environment variable
3. Standard library directories
4. Site-packages (third-party)

Can modify sys.path to add custom paths
""")


# Q4: What's the difference between module and package?
print("\n--- Q4: Module vs Package ---")
print("""
Module: Single .py file
Package: Directory with __init__.py containing modules
""")


# Q5: How to avoid circular imports?
print("\n--- Q5: Circular Imports ---")
print("""
Solutions:
1. Restructure code to avoid circular dependencies
2. Use lazy imports (import inside functions)
3. Move imports to end of file
4. Use importlib for dynamic imports
""")


# ============================================================================
# 9. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("9. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. IMPORT STATEMENTS:
   - import module: Import entire module
   - from module import item: Import specific item
   - import module as alias: Import with alias
   - Follow PEP 8 import order

2. MODULE SEARCH PATH:
   - Python searches sys.path
   - Current directory first
   - Can modify sys.path
   - Use PYTHONPATH environment variable

3. PACKAGES:
   - Directory with __init__.py
   - Can contain modules and subpackages
   - __init__.py initializes package
   - Can define __all__ for exports

4. STANDARD LIBRARY:
   - os: Operating system interface
   - sys: System parameters
   - datetime: Date and time
   - json: JSON data handling
   - csv: CSV file handling
   - math: Mathematical functions
   - random: Random numbers
   - collections: Specialized containers
   - itertools: Iterator functions
   - re: Regular expressions

5. BEST PRACTICES:
   - Use specific imports when possible
   - Avoid 'from module import *'
   - Handle ImportError for optional modules
   - Follow PEP 8 import order
   - Use __all__ in packages

6. COMMON PATTERNS:
   - Conditional imports for optional dependencies
   - Lazy imports for performance
   - Safe imports with try-except
   - Dynamic imports with __import__

7. MODULE ATTRIBUTES:
   - __name__: Module name
   - __file__: Module file path
   - __doc__: Module documentation
   - __all__: Public API definition
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
    
    # Cleanup
    for file in ["example.json", "example.csv"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")
