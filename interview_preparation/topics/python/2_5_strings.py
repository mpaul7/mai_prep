"""
Python Strings - Interview Preparation
Topic 2.5: Strings

This module covers:
- String Methods: upper, lower, strip, split, join, replace, find, index
- String Formatting: f-strings, .format(), % formatting
- String Checking: isdigit, isalpha, isalnum, isspace, startswith, endswith
- String Manipulation: Slicing, concatenation, repetition
- Regular Expressions: re module basics (match, search, findall, sub)
"""

# ============================================================================
# 1. STRING METHODS
# ============================================================================

print("=" * 70)
print("1. STRING METHODS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 upper() and lower() - Case Conversion
# ----------------------------------------------------------------------------
print("\n--- upper() and lower() - Case Conversion ---")

text = "Hello World"

# Convert to uppercase
upper_text = text.upper()
print(f"'{text}'.upper() = '{upper_text}'")

# Convert to lowercase
lower_text = text.lower()
print(f"'{text}'.lower() = '{lower_text}'")

# Original string unchanged (strings are immutable)
print(f"Original unchanged: '{text}'")

# Case-insensitive comparison
str1 = "Hello"
str2 = "HELLO"
print(f"Case-insensitive comparison: {str1.lower() == str2.lower()}")


# ----------------------------------------------------------------------------
# 1.2 strip(), lstrip(), rstrip() - Remove Whitespace
# ----------------------------------------------------------------------------
print("\n--- strip(), lstrip(), rstrip() - Remove Whitespace ---")

# strip() - Remove from both ends
text = "  hello world  "
stripped = text.strip()
print(f"'{text}'.strip() = '{stripped}'")

# lstrip() - Remove from left
left_stripped = text.lstrip()
print(f"'{text}'.lstrip() = '{left_stripped}'")

# rstrip() - Remove from right
right_stripped = text.rstrip()
print(f"'{text}'.rstrip() = '{right_stripped}'")

# strip() with specific characters
text = "***hello***"
stripped = text.strip("*")
print(f"'{text}'.strip('*') = '{stripped}'")

# Common use case: cleaning user input
user_input = "  alice@example.com  "
cleaned = user_input.strip()
print(f"Cleaned email: '{cleaned}'")


# ----------------------------------------------------------------------------
# 1.3 split() - Split String into List
# ----------------------------------------------------------------------------
print("\n--- split() - Split String into List ---")

# Basic split (by whitespace)
text = "apple banana cherry"
words = text.split()
print(f"'{text}'.split() = {words}")

# Split by specific delimiter
text = "apple,banana,cherry"
fruits = text.split(",")
print(f"'{text}'.split(',') = {fruits}")

# Split with maxsplit
text = "one,two,three,four"
parts = text.split(",", maxsplit=2)
print(f"'{text}'.split(',', maxsplit=2) = {parts}")

# Split by multiple characters
text = "apple::banana::cherry"
fruits = text.split("::")
print(f"'{text}'.split('::') = {fruits}")

# Split lines
multiline = "line1\nline2\nline3"
lines = multiline.split("\n")
print(f"Split by newline: {lines}")

# rsplit() - Split from right
text = "one.two.three.four"
parts = text.rsplit(".", maxsplit=1)
print(f"'{text}'.rsplit('.', maxsplit=1) = {parts}")


# ----------------------------------------------------------------------------
# 1.4 join() - Join Iterable into String
# ----------------------------------------------------------------------------
print("\n--- join() - Join Iterable into String ---")

# Join list of strings
words = ["apple", "banana", "cherry"]
joined = ", ".join(words)
print(f"', '.join({words}) = '{joined}'")

# Join with different separator
joined = "-".join(words)
print(f"'-'.join({words}) = '{joined}'")

# Join characters
chars = ["h", "e", "l", "l", "o"]
word = "".join(chars)
print(f"''.join({chars}) = '{word}'")

# Join with newline
lines = ["line1", "line2", "line3"]
text = "\n".join(lines)
print(f"Joined with newline:\n{text}")

# Common pattern: Building strings
parts = ["Hello", "World"]
result = " ".join(parts)
print(f"Built string: '{result}'")


# ----------------------------------------------------------------------------
# 1.5 replace() - Replace Substrings
# ----------------------------------------------------------------------------
print("\n--- replace() - Replace Substrings ---")

text = "Hello World"

# Replace substring
replaced = text.replace("World", "Python")
print(f"'{text}'.replace('World', 'Python') = '{replaced}'")

# Replace with count limit
text = "apple apple apple"
replaced = text.replace("apple", "orange", 2)
print(f"'{text}'.replace('apple', 'orange', 2) = '{replaced}'")

# Replace all occurrences (default)
text = "spam spam spam"
replaced = text.replace("spam", "eggs")
print(f"'{text}'.replace('spam', 'eggs') = '{replaced}'")

# Remove characters (replace with empty string)
text = "hello-world"
cleaned = text.replace("-", "")
print(f"'{text}'.replace('-', '') = '{cleaned}'")


# ----------------------------------------------------------------------------
# 1.6 find() - Find Substring Position
# ----------------------------------------------------------------------------
print("\n--- find() - Find Substring Position ---")

text = "Hello World"

# Find substring (returns index)
index = text.find("World")
print(f"'{text}'.find('World') = {index}")

# Find from specific position
index = text.find("l", 3)  # Start searching from index 3
print(f"'{text}'.find('l', 3) = {index}")

# Find with range
index = text.find("l", 0, 5)  # Search between indices 0 and 5
print(f"'{text}'.find('l', 0, 5) = {index}")

# Returns -1 if not found
index = text.find("Python")
print(f"'{text}'.find('Python') = {index}")  # -1

# rfind() - Find from right
text = "hello world hello"
index = text.rfind("hello")
print(f"'{text}'.rfind('hello') = {index}")


# ----------------------------------------------------------------------------
# 1.7 index() - Find Substring Position (Raises Error)
# ----------------------------------------------------------------------------
print("\n--- index() - Find Substring Position (Raises Error) ---")

text = "Hello World"

# index() works like find() but raises ValueError if not found
index = text.index("World")
print(f"'{text}'.index('World') = {index}")

# ValueError if not found
# index = text.index("Python")  # ValueError: substring not found

# Use find() when you want -1, index() when you want exception
# find() is safer for checking existence
if text.find("World") != -1:
    print("'World' found in text")

# rindex() - Find from right
text = "hello world hello"
index = text.rindex("hello")
print(f"'{text}'.rindex('hello') = {index}")


# ----------------------------------------------------------------------------
# 1.8 Other Useful String Methods
# ----------------------------------------------------------------------------
print("\n--- Other Useful String Methods ---")

text = "Hello World"

# capitalize() - First character uppercase, rest lowercase
print(f"'{text}'.capitalize() = '{text.capitalize()}'")

# title() - First letter of each word uppercase
print(f"'{text}'.title() = '{text.title()}'")

# swapcase() - Swap case
print(f"'{text}'.swapcase() = '{text.swapcase()}'")

# count() - Count occurrences
text = "hello world hello"
count = text.count("hello")
print(f"'{text}'.count('hello') = {count}")

# center(), ljust(), rjust() - Alignment
text = "hello"
print(f"'{text}'.center(10) = '{text.center(10)}'")
print(f"'{text}'.ljust(10) = '{text.ljust(10)}'")
print(f"'{text}'.rjust(10) = '{text.rjust(10)}'")


# ============================================================================
# 2. STRING FORMATTING
# ============================================================================

print("\n" + "=" * 70)
print("2. STRING FORMATTING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 f-strings (Python 3.6+) - Recommended
# ----------------------------------------------------------------------------
print("\n--- f-strings (Python 3.6+) - Recommended ---")

name = "Alice"
age = 25

# Basic f-string
message = f"My name is {name} and I am {age} years old"
print(message)

# Expressions in f-strings
x = 10
y = 20
result = f"The sum of {x} and {y} is {x + y}"
print(result)

# Formatting numbers
pi = 3.14159
print(f"Pi to 2 decimals: {pi:.2f}")
print(f"Pi to 4 decimals: {pi:.4f}")

# Formatting integers
number = 42
print(f"Zero-padded: {number:05d}")
print(f"With comma separator: {number:,}")

# Formatting strings
text = "hello"
print(f"Right-aligned: '{text:>10}'")
print(f"Left-aligned: '{text:<10}'")
print(f"Center-aligned: '{text:^10}'")

# Date formatting
from datetime import datetime
now = datetime.now()
print(f"Current date: {now:%Y-%m-%d %H:%M:%S}")


# ----------------------------------------------------------------------------
# 2.2 .format() Method
# ----------------------------------------------------------------------------
print("\n--- .format() Method ---")

name = "Alice"
age = 25

# Positional arguments
message = "My name is {} and I am {} years old".format(name, age)
print(message)

# Named arguments
message = "My name is {name} and I am {age} years old".format(name=name, age=age)
print(message)

# Indexed arguments
message = "My name is {0} and I am {1} years old".format(name, age)
print(message)

# Formatting numbers
pi = 3.14159
print("Pi to 2 decimals: {:.2f}".format(pi))
print("Pi to 4 decimals: {:.4f}".format(pi))

# Multiple format specifiers
print("Number: {:05d}, Float: {:.2f}".format(42, 3.14159))


# ----------------------------------------------------------------------------
# 2.3 % Formatting (Old Style)
# ----------------------------------------------------------------------------
print("\n--- % Formatting (Old Style) ---")

name = "Alice"
age = 25

# Basic % formatting
message = "My name is %s and I am %d years old" % (name, age)
print(message)

# Format specifiers
pi = 3.14159
print("Pi to 2 decimals: %.2f" % pi)
print("Pi to 4 decimals: %.4f" % pi)

# Dictionary formatting
data = {"name": "Alice", "age": 25}
message = "My name is %(name)s and I am %(age)d years old" % data
print(message)

# Note: f-strings are preferred in Python 3.6+, but % formatting still works


# ----------------------------------------------------------------------------
# 2.4 Formatting Comparison
# ----------------------------------------------------------------------------
print("\n--- Formatting Comparison ---")

name = "Alice"
age = 25
score = 95.5

# f-string (recommended)
message1 = f"{name} is {age} years old and scored {score:.1f}"

# .format()
message2 = "{} is {} years old and scored {:.1f}".format(name, age, score)

# % formatting
message3 = "%s is %d years old and scored %.1f" % (name, age, score)

print(f"f-string: {message1}")
print(f".format(): {message2}")
print(f"% format: {message3}")


# ============================================================================
# 3. STRING CHECKING METHODS
# ============================================================================

print("\n" + "=" * 70)
print("3. STRING CHECKING METHODS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 isdigit() - Check if All Characters are Digits
# ----------------------------------------------------------------------------
print("\n--- isdigit() - Check if All Characters are Digits ---")

# All digits
print(f"'123'.isdigit() = {'123'.isdigit()}")
print(f"'456789'.isdigit() = {'456789'.isdigit()}")

# Contains non-digits
print(f"'123abc'.isdigit() = {'123abc'.isdigit()}")
print(f"'12.34'.isdigit() = {'12.34'.isdigit()}")  # False (contains '.')

# Empty string
print(f"''.isdigit() = {''.isdigit()}")  # False

# Common use case: Validate numeric input
def is_numeric(text):
    """Check if text is numeric."""
    return text.isdigit()

print(f"is_numeric('123') = {is_numeric('123')}")
print(f"is_numeric('12.34') = {is_numeric('12.34')}")


# ----------------------------------------------------------------------------
# 3.2 isalpha() - Check if All Characters are Letters
# ----------------------------------------------------------------------------
print("\n--- isalpha() - Check if All Characters are Letters ---")

# All letters
print(f"'hello'.isalpha() = {'hello'.isalpha()}")
print(f"'WORLD'.isalpha() = {'WORLD'.isalpha()}")

# Contains non-letters
print(f"'hello123'.isalpha() = {'hello123'.isalpha()}")
print(f"'hello world'.isalpha() = {'hello world'.isalpha()}")  # False (space)

# Empty string
print(f"''.isalpha() = {''.isalpha()}")  # False


# ----------------------------------------------------------------------------
# 3.3 isalnum() - Check if All Characters are Alphanumeric
# ----------------------------------------------------------------------------
print("\n--- isalnum() - Check if All Characters are Alphanumeric ---")

# Alphanumeric
print(f"'hello123'.isalnum() = {'hello123'.isalnum()}")
print(f"'ABC123'.isalnum() = {'ABC123'.isalnum()}")

# Contains non-alphanumeric
print(f"'hello 123'.isalnum() = {'hello 123'.isalnum()}")  # False (space)
print(f"'hello-123'.isalnum() = {'hello-123'.isalnum()}")  # False (hyphen)

# Empty string
print(f"''.isalnum() = {''.isalnum()}")  # False


# ----------------------------------------------------------------------------
# 3.4 isspace() - Check if All Characters are Whitespace
# ----------------------------------------------------------------------------
print("\n--- isspace() - Check if All Characters are Whitespace ---")

# Whitespace only
print(f"'   '.isspace() = {'   '.isspace()}")
print(f"'\\n\\t'.isspace() = {'\n\t'.isspace()}")

# Contains non-whitespace
print(f"'hello'.isspace() = {'hello'.isspace()}")
print(f"'  hello  '.isspace() = {'  hello  '.isspace()}")  # False

# Empty string
print(f"''.isspace() = {''.isspace()}")  # False


# ----------------------------------------------------------------------------
# 3.5 startswith() - Check if String Starts with Prefix
# ----------------------------------------------------------------------------
print("\n--- startswith() - Check if String Starts with Prefix ---")

text = "Hello World"

# Single prefix
print(f"'{text}'.startswith('Hello') = {text.startswith('Hello')}")
print(f"'{text}'.startswith('World') = {text.startswith('World')}")

# Multiple prefixes (tuple)
print(f"'{text}'.startswith(('Hello', 'Hi')) = {text.startswith(('Hello', 'Hi'))}")

# With start position
print(f"'{text}'.startswith('World', 6) = {text.startswith('World', 6)}")

# With start and end
print(f"'{text}'.startswith('Hello', 0, 5) = {text.startswith('Hello', 0, 5)}")


# ----------------------------------------------------------------------------
# 3.6 endswith() - Check if String Ends with Suffix
# ----------------------------------------------------------------------------
print("\n--- endswith() - Check if String Ends with Suffix ---")

text = "Hello World"

# Single suffix
print(f"'{text}'.endswith('World') = {text.endswith('World')}")
print(f"'{text}'.endswith('Hello') = {text.endswith('Hello')}")

# Multiple suffixes (tuple)
print(f"'{text}'.endswith(('World', 'Python')) = {text.endswith(('World', 'Python'))}")

# With start position
print(f"'{text}'.endswith('Hello', 0, 5) = {text.endswith('Hello', 0, 5)}")

# Common use case: Check file extension
filename = "document.pdf"
print(f"Is PDF: {filename.endswith('.pdf')}")
print(f"Is document: {filename.endswith(('.pdf', '.doc', '.docx'))}")


# ----------------------------------------------------------------------------
# 3.7 Other Checking Methods
# ----------------------------------------------------------------------------
print("\n--- Other Checking Methods ---")

# islower() - All lowercase
print(f"'hello'.islower() = {'hello'.islower()}")
print(f"'Hello'.islower() = {'Hello'.islower()}")

# isupper() - All uppercase
print(f"'HELLO'.isupper() = {'HELLO'.isupper()}")
print(f"'Hello'.isupper() = {'Hello'.isupper()}")

# istitle() - Title case
print(f"'Hello World'.istitle() = {'Hello World'.istitle()}")
print(f"'hello world'.istitle() = {'hello world'.istitle()}")


# ============================================================================
# 4. STRING MANIPULATION
# ============================================================================

print("\n" + "=" * 70)
print("4. STRING MANIPULATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Slicing
# ----------------------------------------------------------------------------
print("\n--- Slicing ---")

text = "Hello World"

# Basic slicing
print(f"'{text}'[0:5] = '{text[0:5]}'")  # First 5 characters
print(f"'{text}'[6:] = '{text[6:]}'")  # From index 6 to end
print(f"'{text}'[:5] = '{text[:5]}'")  # First 5 characters
print(f"'{text}'[-5:] = '{text[-5:]}'")  # Last 5 characters

# Negative indices
print(f"'{text}'[-1] = '{text[-1]}'")  # Last character
print(f"'{text}'[-5:-1] = '{text[-5:-1]}'")  # Characters from -5 to -1

# Step
print(f"'{text}'[::2] = '{text[::2]}'")  # Every second character
print(f"'{text}'[::-1] = '{text[::-1]}'")  # Reverse string

# More slicing examples
text = "Python"
print(f"'{text}'[1:4] = '{text[1:4]}'")  # 'yth'
print(f"'{text}'[::2] = '{text[::2]}'")  # 'Pto'


# ----------------------------------------------------------------------------
# 4.2 Concatenation
# ----------------------------------------------------------------------------
print("\n--- Concatenation ---")

# Using + operator
str1 = "Hello"
str2 = "World"
result = str1 + " " + str2
print(f"'{str1}' + ' ' + '{str2}' = '{result}'")

# Multiple concatenations
result = "a" + "b" + "c"
print(f"'a' + 'b' + 'c' = '{result}'")

# Concatenation with variables
name = "Alice"
greeting = "Hello, " + name + "!"
print(greeting)

# Using join() for multiple strings (more efficient)
words = ["Hello", "World", "Python"]
result = " ".join(words)
print(f"join(): '{result}'")

# join() is more efficient than + for many strings
# Avoid: result = ""; for word in words: result += word  # Slow!


# ----------------------------------------------------------------------------
# 4.3 Repetition
# ----------------------------------------------------------------------------
print("\n--- Repetition ---")

# Using * operator
text = "Hello"
repeated = text * 3
print(f"'{text}' * 3 = '{repeated}'")

# Repetition with spaces
separator = "-" * 20
print(f"Separator: '{separator}'")

# Building patterns
pattern = "*" * 5
print(f"Pattern: '{pattern}'")

# Repetition with numbers
result = "0" * 5
print(f"'0' * 5 = '{result}'")


# ----------------------------------------------------------------------------
# 4.4 String Immutability
# ----------------------------------------------------------------------------
print("\n--- String Immutability ---")

# Strings are immutable - cannot modify in place
text = "Hello"
print(f"Original: '{text}'")

# These create new strings, don't modify original
upper_text = text.upper()
print(f"After upper(): '{text}' (unchanged)")
print(f"New string: '{upper_text}'")

# Cannot modify character
# text[0] = 'h'  # TypeError: 'str' object does not support item assignment

# To "modify", create new string
text = "H" + text[1:]  # Replace first character
print(f"Modified: '{text}'")


# ----------------------------------------------------------------------------
# 4.5 String Operations Examples
# ----------------------------------------------------------------------------
print("\n--- String Operations Examples ---")

# Reverse string
text = "Python"
reversed_text = text[::-1]
print(f"Reverse of '{text}': '{reversed_text}'")

# Extract substring
text = "Hello World"
substring = text[6:11]
print(f"Substring: '{substring}'")

# Check if substring exists
text = "Hello World"
if "World" in text:
    print("'World' found in text")

# Count occurrences
text = "hello world hello"
count = text.count("hello")
print(f"Count of 'hello': {count}")


# ============================================================================
# 5. REGULAR EXPRESSIONS (re module)
# ============================================================================

print("\n" + "=" * 70)
print("5. REGULAR EXPRESSIONS (re module)")
print("=" * 70)

import re

# ----------------------------------------------------------------------------
# 5.1 re.match() - Match at Beginning
# ----------------------------------------------------------------------------
print("\n--- re.match() - Match at Beginning ---")

# match() checks if pattern matches at the beginning of string
text = "Hello World"
pattern = r"Hello"

# Match object if found, None if not
match = re.match(pattern, text)
if match:
    print(f"Match found: '{match.group()}'")
    print(f"Match span: {match.span()}")

# No match (pattern not at beginning)
match = re.match(r"World", text)
if match:
    print("Match found")
else:
    print("No match (pattern not at beginning)")

# Match with groups
text = "2024-01-15"
pattern = r"(\d{4})-(\d{2})-(\d{2})"
match = re.match(pattern, text)
if match:
    print(f"Full match: '{match.group()}'")
    print(f"Groups: {match.groups()}")
    print(f"Year: {match.group(1)}, Month: {match.group(2)}, Day: {match.group(3)}")


# ----------------------------------------------------------------------------
# 5.2 re.search() - Search Anywhere
# ----------------------------------------------------------------------------
print("\n--- re.search() - Search Anywhere ---")

# search() finds first occurrence anywhere in string
text = "Hello World"
pattern = r"World"

match = re.search(pattern, text)
if match:
    print(f"Found: '{match.group()}' at position {match.start()}-{match.end()}")

# Search for email pattern
text = "Contact me at alice@example.com for details"
pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
match = re.search(pattern, text)
if match:
    print(f"Email found: '{match.group()}'")

# Search with groups
text = "Phone: 123-456-7890"
pattern = r"(\d{3})-(\d{3})-(\d{4})"
match = re.search(pattern, text)
if match:
    print(f"Phone: {match.group(1)}-{match.group(2)}-{match.group(3)}")


# ----------------------------------------------------------------------------
# 5.3 re.findall() - Find All Matches
# ----------------------------------------------------------------------------
print("\n--- re.findall() - Find All Matches ---")

# findall() returns list of all matches
text = "apple banana apple cherry apple"
pattern = r"apple"

matches = re.findall(pattern, text)
print(f"All matches: {matches}")

# Find all digits
text = "I have 3 apples and 5 oranges"
pattern = r"\d+"
numbers = re.findall(pattern, text)
print(f"Numbers found: {numbers}")

# Find all words
text = "Hello World Python"
pattern = r"\w+"
words = re.findall(pattern, text)
print(f"Words found: {words}")

# Find all with groups
text = "Date: 2024-01-15, Date: 2024-02-20"
pattern = r"(\d{4})-(\d{2})-(\d{2})"
matches = re.findall(pattern, text)
print(f"All dates: {matches}")  # Returns list of tuples


# ----------------------------------------------------------------------------
# 5.4 re.sub() - Substitute/Replace
# ----------------------------------------------------------------------------
print("\n--- re.sub() - Substitute/Replace ---")

# sub() replaces all occurrences of pattern
text = "Hello World Hello"
pattern = r"Hello"
replacement = "Hi"
result = re.sub(pattern, replacement, text)
print(f"Original: '{text}'")
print(f"After sub: '{result}'")

# Replace with count limit
result = re.sub(pattern, replacement, text, count=1)
print(f"After sub(count=1): '{result}'")

# Replace digits
text = "I have 3 apples and 5 oranges"
result = re.sub(r"\d+", "X", text)
print(f"After replacing digits: '{result}'")

# Replace with function
def double_number(match):
    """Double the matched number."""
    num = int(match.group())
    return str(num * 2)

text = "I have 3 apples and 5 oranges"
result = re.sub(r"\d+", double_number, text)
print(f"After doubling numbers: '{result}'")

# Format phone numbers
text = "Call 1234567890 or 9876543210"
pattern = r"(\d{3})(\d{3})(\d{4})"
replacement = r"(\1) \2-\3"
result = re.sub(pattern, replacement, text)
print(f"Formatted phones: '{result}'")


# ----------------------------------------------------------------------------
# 5.5 Common Regex Patterns
# ----------------------------------------------------------------------------
print("\n--- Common Regex Patterns ---")

# Email pattern
email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
text = "Contact alice@example.com or bob@test.org"
emails = re.findall(email_pattern, text)
print(f"Emails found: {emails}")

# Phone number pattern
phone_pattern = r"\d{3}-\d{3}-\d{4}"
text = "Call 123-456-7890 or 987-654-3210"
phones = re.findall(phone_pattern, text)
print(f"Phones found: {phones}")

# Date pattern
date_pattern = r"\d{4}-\d{2}-\d{2}"
text = "Dates: 2024-01-15, 2024-02-20, 2024-03-25"
dates = re.findall(date_pattern, text)
print(f"Dates found: {dates}")

# Word boundaries
text = "The cat sat on the mat"
pattern = r"\bcat\b"  # Match 'cat' as whole word
match = re.search(pattern, text)
if match:
    print(f"Found word 'cat': '{match.group()}'")


# ----------------------------------------------------------------------------
# 5.6 Regex Flags
# ----------------------------------------------------------------------------
print("\n--- Regex Flags ---")

# re.IGNORECASE (or re.I) - Case insensitive
text = "Hello World"
pattern = r"hello"
match = re.search(pattern, text, re.IGNORECASE)
if match:
    print(f"Case-insensitive match: '{match.group()}'")

# re.MULTILINE (or re.M) - ^ and $ match line boundaries
text = "line1\nline2\nline3"
pattern = r"^line"
matches = re.findall(pattern, text, re.MULTILINE)
print(f"Matches with MULTILINE: {matches}")

# re.DOTALL (or re.S) - . matches newline
text = "hello\nworld"
pattern = r"hello.world"
match = re.search(pattern, text, re.DOTALL)
if match:
    print(f"Match with DOTALL: '{match.group()}'")


# ============================================================================
# 6. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Reverse words in string
print("\n--- Exercise 1: Reverse Words ---")
def reverse_words(text):
    """Reverse order of words in string."""
    words = text.split()
    return " ".join(reversed(words))

text = "Hello World Python"
print(f"'{text}' -> '{reverse_words(text)}'")


# Exercise 2: Count words
print("\n--- Exercise 2: Count Words ---")
def count_words(text):
    """Count number of words in string."""
    return len(text.split())

text = "Hello World Python Programming"
print(f"Word count in '{text}': {count_words(text)}")


# Exercise 3: Check if palindrome
print("\n--- Exercise 3: Check Palindrome ---")
def is_palindrome(text):
    """Check if string is palindrome."""
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

print(f"'racecar' is palindrome: {is_palindrome('racecar')}")
print(f"'hello' is palindrome: {is_palindrome('hello')}")
print(f"'A man a plan a canal Panama' is palindrome: {is_palindrome('A man a plan a canal Panama')}")


# Exercise 4: Extract email addresses
print("\n--- Exercise 4: Extract Emails ---")
import re

def extract_emails(text):
    """Extract email addresses from text."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.findall(pattern, text)

text = "Contact alice@example.com or bob@test.org for info"
emails = extract_emails(text)
print(f"Emails found: {emails}")


# Exercise 5: Format phone numbers
print("\n--- Exercise 5: Format Phone Numbers ---")
import re

def format_phone(text):
    """Format phone numbers to (XXX) XXX-XXXX."""
    pattern = r"(\d{3})(\d{3})(\d{4})"
    replacement = r"(\1) \2-\3"
    return re.sub(pattern, replacement, text)

text = "Call 1234567890 or 9876543210"
formatted = format_phone(text)
print(f"Formatted: '{formatted}'")


# Exercise 6: Remove special characters
print("\n--- Exercise 6: Remove Special Characters ---")
import re

def remove_special(text):
    """Remove special characters, keep only alphanumeric and spaces."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)

text = "Hello, World! How are you?"
cleaned = remove_special(text)
print(f"Cleaned: '{cleaned}'")


# Exercise 7: Capitalize first letter of each word
print("\n--- Exercise 7: Title Case ---")
def title_case(text):
    """Capitalize first letter of each word."""
    return text.title()

text = "hello world python"
print(f"Title case: '{title_case(text)}'")


# Exercise 8: Validate string format
print("\n--- Exercise 8: Validate Format ---")
import re

def validate_email(email):
    """Validate email format."""
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$"
    return bool(re.match(pattern, email))

emails = ["alice@example.com", "invalid.email", "test@domain"]
for email in emails:
    print(f"'{email}' is valid: {validate_email(email)}")


# ============================================================================
# 7. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("7. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What's the difference between find() and index()?
print("\n--- Q1: find() vs index() ---")
print("""
find(): Returns -1 if substring not found
index(): Raises ValueError if substring not found
Use find() when you want to check existence without exception
Use index() when you want exception for missing substring
""")


# Q2: What's the difference between split() and partition()?
print("\n--- Q2: split() vs partition() ---")
print("""
split(): Splits string into list of parts
partition(): Splits into 3 parts: (before, separator, after)
partition() always returns 3 elements, split() can return any number
""")


# Q3: How to reverse a string?
print("\n--- Q3: Reversing String ---")
print("""
Method 1: Slicing - text[::-1] (most Pythonic)
Method 2: reversed() - ''.join(reversed(text))
Method 3: Loop - build new string character by character
Slicing is most efficient and readable
""")


# Q4: What's the best way to format strings?
print("\n--- Q4: String Formatting ---")
print("""
Python 3.6+: f-strings (recommended)
- Fast, readable, supports expressions
- f"Hello {name}"

Python 3+: .format() method
- Flexible, supports named arguments
- "Hello {}".format(name)

Old style: % formatting
- Still works but not recommended
- "Hello %s" % name
""")


# Q5: Are strings mutable?
print("\n--- Q5: String Mutability ---")
print("""
No, strings are immutable in Python
Operations like upper(), replace() create new strings
Original string is never modified
This ensures string safety and allows optimizations
""")


# Q6: When to use regex vs string methods?
print("\n--- Q6: Regex vs String Methods ---")
print("""
Use string methods for:
- Simple operations (split, replace, find)
- Fixed patterns
- Better performance for simple cases

Use regex for:
- Complex patterns
- Pattern matching
- Flexible matching rules
- When string methods are insufficient
""")


# Q7: What's the difference between match() and search()?
print("\n--- Q7: match() vs search() ---")
print("""
match(): Checks if pattern matches at beginning of string
search(): Finds first occurrence anywhere in string
findall(): Finds all occurrences in string
Use match() for validation, search() for finding patterns
""")


# ============================================================================
# 8. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("8. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. STRING METHODS:
   - upper(), lower(): Case conversion
   - strip(), lstrip(), rstrip(): Remove whitespace
   - split(): Split into list
   - join(): Join iterable into string
   - replace(): Replace substrings
   - find(), index(): Find substring position
   - count(): Count occurrences

2. STRING FORMATTING:
   - f-strings (Python 3.6+): Recommended, fast, readable
   - .format(): Flexible, supports named arguments
   - % formatting: Old style, still works
   - Use f-strings when possible

3. STRING CHECKING:
   - isdigit(): All digits
   - isalpha(): All letters
   - isalnum(): Alphanumeric
   - isspace(): Whitespace only
   - startswith(), endswith(): Prefix/suffix checking

4. STRING MANIPULATION:
   - Slicing: text[start:stop:step]
   - Concatenation: + operator or join()
   - Repetition: * operator
   - Strings are immutable (operations create new strings)

5. REGULAR EXPRESSIONS:
   - re.match(): Match at beginning
   - re.search(): Find anywhere
   - re.findall(): Find all matches
   - re.sub(): Replace matches
   - Use for complex pattern matching

6. COMMON PATTERNS:
   - Reverse: text[::-1]
   - Split and join: ' '.join(text.split())
   - Remove whitespace: text.strip()
   - Check substring: 'sub' in text
   - Count words: len(text.split())

7. BEST PRACTICES:
   - Use f-strings for formatting (Python 3.6+)
   - Use join() for concatenating multiple strings
   - Use find() for safe substring checking
   - Use regex for complex patterns
   - Remember strings are immutable
   - Use string methods when possible (faster than regex for simple cases)

8. PERFORMANCE:
   - join() is faster than + for multiple strings
   - String methods are faster than regex for simple operations
   - Use regex only when necessary
   - Membership testing (in) is O(n) for strings
""")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
