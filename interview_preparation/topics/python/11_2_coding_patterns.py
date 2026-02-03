"""
Python Coding Patterns - Interview Preparation
Topic 11.2: Coding Patterns

This module covers:
- String Manipulation: Parsing, formatting, validation
- Data Processing: Filtering, transforming, aggregating
- Algorithm Implementation: Sorting, searching, counting
- Problem Solving: Breaking down problems, step-by-step approach
"""

from collections import Counter, defaultdict
import re

# ============================================================================
# 1. STRING MANIPULATION PATTERNS
# ============================================================================

print("=" * 70)
print("1. STRING MANIPULATION PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Parsing Patterns
# ----------------------------------------------------------------------------
print("\n--- 1.1 Parsing Patterns ---")

# Pattern: Parse CSV-like string
def parse_csv_line(line):
    """Parse comma-separated values."""
    return [item.strip() for item in line.split(',')]

line = "apple, banana, cherry, date"
parsed = parse_csv_line(line)
print(f"Parsed CSV: {parsed}")

# Pattern: Parse key-value pairs
def parse_key_value(text, separator='='):
    """Parse key=value pairs."""
    result = {}
    for pair in text.split(','):
        if separator in pair:
            key, value = pair.split(separator, 1)
            result[key.strip()] = value.strip()
    return result

text = "name=Alice, age=25, city=New York"
parsed = parse_key_value(text)
print(f"Parsed key-value: {parsed}")

# Pattern: Extract numbers from string
def extract_numbers(text):
    """Extract all numbers from string."""
    return [int(x) for x in re.findall(r'\d+', text)]

text = "I have 5 apples and 3 oranges, total 8 fruits"
numbers = extract_numbers(text)
print(f"Extracted numbers: {numbers}")

# Pattern: Parse nested structures (simple)
def parse_nested(text, open_char='(', close_char=')'):
    """Parse nested parentheses."""
    stack = []
    result = []
    current = []
    
    for char in text:
        if char == open_char:
            stack.append(current)
            current = []
        elif char == close_char:
            if stack:
                result.append(''.join(current))
                current = stack.pop()
        else:
            current.append(char)
    
    return result

text = "(hello (world) (python))"
parsed = parse_nested(text)
print(f"Parsed nested: {parsed}")


# ----------------------------------------------------------------------------
# 1.2 Formatting Patterns
# ----------------------------------------------------------------------------
print("\n--- 1.2 Formatting Patterns ---")

# Pattern: Format with padding
def format_with_padding(text, width, align='left', fill=' '):
    """Format string with padding."""
    if align == 'left':
        return text.ljust(width, fill)
    elif align == 'right':
        return text.rjust(width, fill)
    else:  # center
        return text.center(width, fill)

print(f"Left: '{format_with_padding('hello', 10)}'")
print(f"Right: '{format_with_padding('hello', 10, 'right')}'")
print(f"Center: '{format_with_padding('hello', 10, 'center')}'")

# Pattern: Format numbers
def format_number(num, decimals=2):
    """Format number with specified decimals."""
    return f"{num:.{decimals}f}"

print(f"Formatted: {format_number(3.14159, 2)}")
print(f"Formatted: {format_number(1000, 0)}")

# Pattern: Format table-like output
def format_table(data, headers):
    """Format data as table."""
    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Format header
    header_row = ' | '.join(h.ljust(w) for h, w in zip(headers, widths))
    separator = '-' * len(header_row)
    
    # Format rows
    rows = [' | '.join(str(cell).ljust(w) for cell, w in zip(row, widths)) 
            for row in data]
    
    return '\n'.join([header_row, separator] + rows)

data = [['Alice', 25, 'NYC'], ['Bob', 30, 'LA']]
headers = ['Name', 'Age', 'City']
print(f"\nTable:\n{format_table(data, headers)}")


# ----------------------------------------------------------------------------
# 1.3 Validation Patterns
# ----------------------------------------------------------------------------
print("\n--- 1.3 Validation Patterns ---")

# Pattern: Validate email
def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

emails = ["alice@example.com", "invalid.email", "test@domain.co.uk"]
for email in emails:
    print(f"{email}: {validate_email(email)}")

# Pattern: Validate phone number
def validate_phone(phone):
    """Validate phone number format."""
    # Remove common separators
    cleaned = re.sub(r'[-()\s]', '', phone)
    # Check if all digits and correct length
    return cleaned.isdigit() and len(cleaned) == 10

phones = ["123-456-7890", "(123) 456-7890", "1234567890", "123"]
for phone in phones:
    print(f"{phone}: {validate_phone(phone)}")

# Pattern: Validate password strength
def validate_password(password):
    """Validate password strength."""
    checks = {
        'length': len(password) >= 8,
        'has_upper': any(c.isupper() for c in password),
        'has_lower': any(c.islower() for c in password),
        'has_digit': any(c.isdigit() for c in password),
        'has_special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    }
    return checks

password = "MyP@ssw0rd"
checks = validate_password(password)
print(f"Password checks: {checks}")
print(f"Valid: {all(checks.values())}")


# ============================================================================
# 2. DATA PROCESSING PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("2. DATA PROCESSING PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Filtering Patterns
# ----------------------------------------------------------------------------
print("\n--- 2.1 Filtering Patterns ---")

# Pattern: Filter by condition
def filter_even(numbers):
    """Filter even numbers."""
    return [n for n in numbers if n % 2 == 0]

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = filter_even(numbers)
print(f"Even numbers: {evens}")

# Pattern: Filter by multiple conditions
def filter_range(numbers, min_val, max_val):
    """Filter numbers in range."""
    return [n for n in numbers if min_val <= n <= max_val]

filtered = filter_range(numbers, 3, 7)
print(f"Numbers in range [3, 7]: {filtered}")

# Pattern: Filter unique values
def filter_unique(items):
    """Filter to unique values preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

items = [1, 2, 2, 3, 3, 3, 4, 5, 5]
unique = filter_unique(items)
print(f"Unique items: {unique}")

# Pattern: Filter by type
def filter_by_type(items, target_type):
    """Filter items by type."""
    return [item for item in items if isinstance(item, target_type)]

mixed = [1, "hello", 2.5, "world", 3, [1, 2]]
strings = filter_by_type(mixed, str)
print(f"Strings: {strings}")


# ----------------------------------------------------------------------------
# 2.2 Transforming Patterns
# ----------------------------------------------------------------------------
print("\n--- 2.2 Transforming Patterns ---")

# Pattern: Map transformation
def square_numbers(numbers):
    """Square all numbers."""
    return [n ** 2 for n in numbers]

squared = square_numbers([1, 2, 3, 4, 5])
print(f"Squared: {squared}")

# Pattern: Transform with condition
def transform_conditional(numbers):
    """Double evens, triple odds."""
    return [n * 2 if n % 2 == 0 else n * 3 for n in numbers]

transformed = transform_conditional([1, 2, 3, 4, 5])
print(f"Transformed: {transformed}")

# Pattern: Normalize data
def normalize_scores(scores, max_score=100):
    """Normalize scores to 0-1 range."""
    return [score / max_score for score in scores]

scores = [75, 80, 90, 65, 95]
normalized = normalize_scores(scores)
print(f"Normalized: {normalized}")

# Pattern: Transform nested structures
def flatten_nested(nested_list):
    """Flatten nested list."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_nested(item))
        else:
            result.append(item)
    return result

nested = [[1, 2], [3, [4, 5]], 6]
flattened = flatten_nested(nested)
print(f"Flattened: {flattened}")


# ----------------------------------------------------------------------------
# 2.3 Aggregating Patterns
# ----------------------------------------------------------------------------
print("\n--- 2.3 Aggregating Patterns ---")

# Pattern: Count occurrences
def count_items(items):
    """Count occurrences of each item."""
    return Counter(items)

items = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
counts = count_items(items)
print(f"Counts: {counts}")

# Pattern: Group by key
def group_by_key(data, key_func):
    """Group items by key function."""
    grouped = defaultdict(list)
    for item in data:
        key = key_func(item)
        grouped[key].append(item)
    return dict(grouped)

students = [
    {'name': 'Alice', 'grade': 'A'},
    {'name': 'Bob', 'grade': 'B'},
    {'name': 'Charlie', 'grade': 'A'},
    {'name': 'David', 'grade': 'B'}
]
grouped = group_by_key(students, lambda x: x['grade'])
print(f"Grouped by grade: {grouped}")

# Pattern: Calculate statistics
def calculate_stats(numbers):
    """Calculate basic statistics."""
    if not numbers:
        return None
    
    return {
        'count': len(numbers),
        'sum': sum(numbers),
        'mean': sum(numbers) / len(numbers),
        'min': min(numbers),
        'max': max(numbers)
    }

stats = calculate_stats([10, 20, 30, 40, 50])
print(f"Statistics: {stats}")

# Pattern: Aggregate with initial value
def aggregate_with_initial(items, initial, func):
    """Aggregate with initial value."""
    result = initial
    for item in items:
        result = func(result, item)
    return result

numbers = [1, 2, 3, 4, 5]
product = aggregate_with_initial(numbers, 1, lambda x, y: x * y)
print(f"Product: {product}")


# ============================================================================
# 3. ALGORITHM IMPLEMENTATION PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("3. ALGORITHM IMPLEMENTATION PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Sorting Patterns
# ----------------------------------------------------------------------------
print("\n--- 3.1 Sorting Patterns ---")

# Pattern: Sort by key
def sort_by_key(items, key_func):
    """Sort items by key function."""
    return sorted(items, key=key_func)

students = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 20},
    {'name': 'Charlie', 'age': 30}
]
sorted_by_age = sort_by_key(students, lambda x: x['age'])
print(f"Sorted by age: {sorted_by_age}")

# Pattern: Sort by multiple keys
def sort_by_multiple_keys(items, key_funcs):
    """Sort by multiple keys."""
    return sorted(items, key=lambda x: tuple(f(x) for f in key_funcs))

# Sort by age, then by name
sorted_multi = sort_by_multiple_keys(
    students,
    [lambda x: x['age'], lambda x: x['name']]
)
print(f"Sorted by age then name: {sorted_multi}")

# Pattern: Custom sort order
def sort_custom_order(items, order):
    """Sort by custom order."""
    order_map = {item: i for i, item in enumerate(order)}
    return sorted(items, key=lambda x: order_map.get(x, len(order)))

colors = ['red', 'blue', 'green', 'red', 'blue']
custom_order = ['blue', 'red', 'green']
sorted_colors = sort_custom_order(colors, custom_order)
print(f"Custom sorted: {sorted_colors}")


# ----------------------------------------------------------------------------
# 3.2 Searching Patterns
# ----------------------------------------------------------------------------
print("\n--- 3.2 Searching Patterns ---")

# Pattern: Linear search
def linear_search(items, target):
    """Linear search for target."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

numbers = [5, 2, 8, 1, 9, 3]
index = linear_search(numbers, 8)
print(f"Found 8 at index: {index}")

# Pattern: Binary search (sorted list)
def binary_search(items, target):
    """Binary search in sorted list."""
    left, right = 0, len(items) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if items[mid] == target:
            return mid
        elif items[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

sorted_numbers = [1, 2, 3, 5, 8, 9]
index = binary_search(sorted_numbers, 5)
print(f"Found 5 at index: {index}")

# Pattern: Find all occurrences
def find_all(items, target):
    """Find all occurrences of target."""
    return [i for i, item in enumerate(items) if item == target]

numbers = [1, 2, 3, 2, 4, 2, 5]
indices = find_all(numbers, 2)
print(f"Found 2 at indices: {indices}")


# ----------------------------------------------------------------------------
# 3.3 Counting Patterns
# ----------------------------------------------------------------------------
print("\n--- 3.3 Counting Patterns ---")

# Pattern: Count with condition
def count_if(items, condition):
    """Count items matching condition."""
    return sum(1 for item in items if condition(item))

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_count = count_if(numbers, lambda x: x % 2 == 0)
print(f"Even count: {even_count}")

# Pattern: Count by category
def count_by_category(items, category_func):
    """Count items by category."""
    counts = defaultdict(int)
    for item in items:
        category = category_func(item)
        counts[category] += 1
    return dict(counts)

words = ['apple', 'banana', 'apricot', 'berry', 'blueberry']
counts = count_by_category(words, lambda x: x[0])
print(f"Counts by first letter: {counts}")

# Pattern: Frequency analysis
def frequency_analysis(text):
    """Analyze character frequency."""
    return Counter(text.lower())

text = "Hello World"
freq = frequency_analysis(text)
print(f"Character frequency: {freq}")


# ============================================================================
# 4. PROBLEM SOLVING APPROACHES
# ============================================================================

print("\n" + "=" * 70)
print("4. PROBLEM SOLVING APPROACHES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Step-by-Step Problem Solving
# ----------------------------------------------------------------------------
print("\n--- 4.1 Step-by-Step Problem Solving ---")

print("""
PROBLEM SOLVING FRAMEWORK:

1. UNDERSTAND THE PROBLEM
   - Read carefully
   - Identify inputs and outputs
   - Understand constraints
   - Ask clarifying questions

2. EXAMPLES & EDGE CASES
   - Work through examples
   - Identify edge cases:
     * Empty inputs
     * Single element
     * Large inputs
     * Boundary values

3. APPROACH
   - Think of brute force first
   - Identify patterns
   - Consider data structures
   - Think about time/space complexity

4. IMPLEMENTATION
   - Start with simple solution
   - Write clean code
   - Use meaningful names
   - Add comments for complex logic

5. TEST & VERIFY
   - Test with examples
   - Test edge cases
   - Verify correctness
   - Check time/space complexity

6. OPTIMIZE (if needed)
   - Identify bottlenecks
   - Consider better algorithms
   - Optimize data structures
   - Trade-offs discussion
""")


# ----------------------------------------------------------------------------
# 4.2 Example: Two Sum Problem
# ----------------------------------------------------------------------------
print("\n--- 4.2 Example: Two Sum Problem ---")
print("""
PROBLEM: Find two numbers that add up to target

STEP 1: UNDERSTAND
- Input: list of numbers, target sum
- Output: indices of two numbers
- Constraints: exactly one solution exists

STEP 2: EXAMPLES
- [2, 7, 11, 15], target=9 → [0, 1] (2+7=9)
- [3, 2, 4], target=6 → [1, 2] (2+4=6)

STEP 3: APPROACHES
- Brute force: O(n²) - check all pairs
- Hash map: O(n) - store complements

STEP 4: IMPLEMENTATION
""")

def two_sum_brute_force(nums, target):
    """Brute force approach - O(n²)."""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum_optimal(nums, target):
    """Optimal approach with hash map - O(n)."""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

nums = [2, 7, 11, 15]
target = 9
print(f"Brute force: {two_sum_brute_force(nums, target)}")
print(f"Optimal: {two_sum_optimal(nums, target)}")


# ----------------------------------------------------------------------------
# 4.3 Example: String Reversal
# ----------------------------------------------------------------------------
print("\n--- 4.3 Example: String Reversal ---")
print("""
PROBLEM: Reverse a string

STEP 1: UNDERSTAND
- Input: string
- Output: reversed string
- Constraints: handle empty string, special characters

STEP 2: EXAMPLES
- "hello" → "olleh"
- "" → ""
- "a" → "a"

STEP 3: APPROACHES
- Slicing: text[::-1] (most Pythonic)
- Built-in: ''.join(reversed(text))
- Loop: build new string

STEP 4: IMPLEMENTATION
""")

def reverse_string_slicing(text):
    """Reverse using slicing - O(n)."""
    return text[::-1]

def reverse_string_builtin(text):
    """Reverse using built-in - O(n)."""
    return ''.join(reversed(text))

def reverse_string_loop(text):
    """Reverse using loop - O(n)."""
    result = []
    for char in text:
        result.insert(0, char)
    return ''.join(result)

text = "hello"
print(f"Slicing: {reverse_string_slicing(text)}")
print(f"Built-in: {reverse_string_builtin(text)}")
print(f"Loop: {reverse_string_loop(text)}")


# ----------------------------------------------------------------------------
# 4.4 Example: Finding Maximum
# ----------------------------------------------------------------------------
print("\n--- 4.4 Example: Finding Maximum ---")
print("""
PROBLEM: Find maximum value in list

STEP 1: UNDERSTAND
- Input: list of numbers
- Output: maximum value
- Edge cases: empty list, single element, all same

STEP 2: APPROACHES
- Built-in: max(numbers)
- Manual: iterate and track maximum
- Reduce: use reduce function

STEP 4: IMPLEMENTATION
""")

def find_max_builtin(numbers):
    """Using built-in max()."""
    if not numbers:
        return None
    return max(numbers)

def find_max_manual(numbers):
    """Manual iteration - O(n)."""
    if not numbers:
        return None
    
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val

numbers = [3, 7, 2, 9, 1, 5]
print(f"Built-in: {find_max_builtin(numbers)}")
print(f"Manual: {find_max_manual(numbers)}")


# ============================================================================
# 5. COMMON PATTERNS SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("5. COMMON PATTERNS SUMMARY")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Two Pointers Pattern
# ----------------------------------------------------------------------------
print("\n--- 5.1 Two Pointers Pattern ---")

def two_pointers_example(arr, target):
    """Find two numbers that sum to target (sorted array)."""
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

sorted_arr = [1, 2, 3, 4, 5, 6, 7]
result = two_pointers_example(sorted_arr, 9)
print(f"Two pointers result: {result}")


# ----------------------------------------------------------------------------
# 5.2 Sliding Window Pattern
# ----------------------------------------------------------------------------
print("\n--- 5.2 Sliding Window Pattern ---")

def sliding_window_max_sum(arr, k):
    """Find maximum sum of subarray of size k."""
    if len(arr) < k:
        return None
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

arr = [1, 4, 2, 10, 2, 3, 1, 0, 20]
result = sliding_window_max_sum(arr, 4)
print(f"Max sum of window size 4: {result}")


# ----------------------------------------------------------------------------
# 5.3 Hash Map Pattern
# ----------------------------------------------------------------------------
print("\n--- 5.3 Hash Map Pattern ---")

def find_duplicates(arr):
    """Find duplicates using hash map."""
    seen = {}
    duplicates = []
    
    for num in arr:
        if num in seen:
            if num not in duplicates:
                duplicates.append(num)
        else:
            seen[num] = True
    
    return duplicates

arr = [1, 2, 3, 2, 4, 3, 5]
duplicates = find_duplicates(arr)
print(f"Duplicates: {duplicates}")


# ----------------------------------------------------------------------------
# 5.4 Prefix Sum Pattern
# ----------------------------------------------------------------------------
print("\n--- 5.4 Prefix Sum Pattern ---")

def prefix_sum(arr):
    """Calculate prefix sum array."""
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix, left, right):
    """Get sum of range using prefix sum."""
    return prefix[right + 1] - prefix[left]

arr = [1, 2, 3, 4, 5]
prefix = prefix_sum(arr)
print(f"Prefix sum: {prefix}")
print(f"Sum [1, 3]: {range_sum(prefix, 1, 3)}")


# ============================================================================
# 6. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("6. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. STRING MANIPULATION:
   - Use string methods for simple operations
   - Use regex for complex patterns
   - Remember strings are immutable
   - Use join() for concatenation

2. DATA PROCESSING:
   - List comprehensions for transformations
   - Counter/defaultdict for counting
   - Filter before transform when possible
   - Use generators for large datasets

3. ALGORITHMS:
   - Know time/space complexity
   - Start with brute force, then optimize
   - Use appropriate data structures
   - Consider trade-offs

4. PROBLEM SOLVING:
   - Understand problem first
   - Work through examples
   - Think out loud
   - Test edge cases
   - Optimize if needed

5. COMMON PATTERNS:
   - Two pointers: sorted arrays, palindromes
   - Sliding window: subarray problems
   - Hash map: fast lookups, counting
   - Prefix sum: range queries
   - Greedy: local optimal choices

6. BEST PRACTICES:
   - Write clean, readable code
   - Use meaningful variable names
   - Handle edge cases
   - Explain your approach
   - Discuss trade-offs
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Coding Patterns Guide Ready!")
    print("=" * 70)
    print("\nPractice these patterns:")
    print("- String manipulation: parsing, formatting, validation")
    print("- Data processing: filtering, transforming, aggregating")
    print("- Algorithms: sorting, searching, counting")
    print("- Problem solving: step-by-step approach")
