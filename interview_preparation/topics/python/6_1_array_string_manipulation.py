"""
Array/String Manipulation - Interview Preparation
Topic 6.1: Array/String Manipulation

This module covers:
- Two Pointers: Finding pairs, palindromes
- Sliding Window: Subarray problems, substring problems
- Prefix Sum: Cumulative sums
- String Parsing: Extracting information from strings
"""

from typing import List, Tuple, Optional
from collections import Counter

# ============================================================================
# 1. TWO POINTERS
# ============================================================================

print("=" * 70)
print("1. TWO POINTERS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Two Pointers - Finding Pairs
# ----------------------------------------------------------------------------
print("\n--- 1.1 Two Pointers - Finding Pairs ---")

def two_sum_sorted(arr: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Find two numbers that sum to target in sorted array.
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return (arr[left], arr[right])
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return None

# Example
arr = [1, 2, 3, 4, 5, 6, 7]
target = 9
result = two_sum_sorted(arr, target)
print(f"Array: {arr}, Target: {target}")
print(f"Two sum result: {result}")


def two_sum_all_pairs(arr: List[int], target: int) -> List[Tuple[int, int]]:
    """
    Find all pairs that sum to target in sorted array.
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    pairs = []
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            pairs.append((arr[left], arr[right]))
            left += 1
            right -= 1
            # Skip duplicates
            while left < right and arr[left] == arr[left - 1]:
                left += 1
            while left < right and arr[right] == arr[right + 1]:
                right -= 1
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return pairs

# Example
arr = [1, 2, 2, 3, 4, 5, 6]
target = 6
result = two_sum_all_pairs(arr, target)
print(f"\nArray: {arr}, Target: {target}")
print(f"All pairs: {result}")


# ----------------------------------------------------------------------------
# 1.2 Two Pointers - Palindromes
# ----------------------------------------------------------------------------
print("\n--- 1.2 Two Pointers - Palindromes ---")

def is_palindrome_two_pointers(s: str) -> bool:
    """
    Check if string is palindrome using two pointers.
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    
    return True

# Example
test_strings = ["racecar", "hello", "a", "abba"]
for s in test_strings:
    result = is_palindrome_two_pointers(s)
    print(f"'{s}' is palindrome: {result}")


def is_palindrome_ignore_case(s: str) -> bool:
    """
    Check palindrome ignoring case and non-alphanumeric.
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

# Example
test_strings = ["A man a plan a canal Panama", "race a car", "Madam"]
for s in test_strings:
    result = is_palindrome_ignore_case(s)
    print(f"'{s}' is palindrome: {result}")


def longest_palindrome_substring(s: str) -> str:
    """
    Find longest palindrome substring using two pointers.
    Time: O(n²), Space: O(1)
    """
    def expand_around_center(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = ""
    
    for i in range(len(s)):
        # Odd length palindromes
        palindrome1 = expand_around_center(i, i)
        # Even length palindromes
        palindrome2 = expand_around_center(i, i + 1)
        
        longest = max(longest, palindrome1, palindrome2, key=len)
    
    return longest

# Example
test_strings = ["babad", "cbbd", "racecar"]
for s in test_strings:
    result = longest_palindrome_substring(s)
    print(f"Longest palindrome in '{s}': '{result}'")


# ----------------------------------------------------------------------------
# 1.3 Two Pointers - Removing Duplicates
# ----------------------------------------------------------------------------
print("\n--- 1.3 Two Pointers - Removing Duplicates ---")

def remove_duplicates_sorted(arr: List[int]) -> int:
    """
    Remove duplicates from sorted array in-place.
    Returns new length.
    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0
    
    write_index = 1
    
    for read_index in range(1, len(arr)):
        if arr[read_index] != arr[read_index - 1]:
            arr[write_index] = arr[read_index]
            write_index += 1
    
    return write_index

# Example
arr = [1, 1, 2, 2, 3, 4, 4, 5]
print(f"Original: {arr}")
new_length = remove_duplicates_sorted(arr)
print(f"After removing duplicates: {arr[:new_length]}")


def move_zeros_to_end(arr: List[int]) -> None:
    """
    Move all zeros to end while maintaining relative order.
    Time: O(n), Space: O(1)
    """
    write_index = 0
    
    # Move non-zero elements to front
    for read_index in range(len(arr)):
        if arr[read_index] != 0:
            arr[write_index] = arr[read_index]
            write_index += 1
    
    # Fill remaining with zeros
    while write_index < len(arr):
        arr[write_index] = 0
        write_index += 1

# Example
arr = [0, 1, 0, 3, 12]
print(f"\nOriginal: {arr}")
move_zeros_to_end(arr)
print(f"After moving zeros: {arr}")


# ============================================================================
# 2. SLIDING WINDOW
# ============================================================================

print("\n" + "=" * 70)
print("2. SLIDING WINDOW")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Fixed Size Sliding Window
# ----------------------------------------------------------------------------
print("\n--- 2.1 Fixed Size Sliding Window ---")

def max_sum_subarray_fixed(arr: List[int], k: int) -> int:
    """
    Find maximum sum of subarray of fixed size k.
    Time: O(n), Space: O(1)
    """
    if len(arr) < k:
        return 0
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Example
arr = [1, 4, 2, 10, 2, 3, 1, 0, 20]
k = 4
result = max_sum_subarray_fixed(arr, k)
print(f"Array: {arr}, Window size: {k}")
print(f"Maximum sum: {result}")


def average_subarray_fixed(arr: List[int], k: int) -> List[float]:
    """
    Find average of all subarrays of size k.
    Time: O(n), Space: O(1)
    """
    if len(arr) < k:
        return []
    
    averages = []
    window_sum = sum(arr[:k])
    averages.append(window_sum / k)
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        averages.append(window_sum / k)
    
    return averages

# Example
arr = [1, 3, 2, 6, -1, 4, 1, 8, 2]
k = 5
result = average_subarray_fixed(arr, k)
print(f"\nArray: {arr}, Window size: {k}")
print(f"Averages: {result}")


# ----------------------------------------------------------------------------
# 2.2 Variable Size Sliding Window
# ----------------------------------------------------------------------------
print("\n--- 2.2 Variable Size Sliding Window ---")

def longest_substring_no_repeat(s: str) -> int:
    """
    Find length of longest substring without repeating characters.
    Time: O(n), Space: O(min(n, m)) where m is charset size
    """
    char_index = {}
    max_length = 0
    start = 0
    
    for end in range(len(s)):
        # If character seen, move start pointer
        if s[end] in char_index and char_index[s[end]] >= start:
            start = char_index[s[end]] + 1
        
        char_index[s[end]] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Example
test_strings = ["abcabcbb", "bbbbb", "pwwkew"]
for s in test_strings:
    result = longest_substring_no_repeat(s)
    print(f"Longest substring in '{s}': {result}")


def min_subarray_sum_target(arr: List[int], target: int) -> int:
    """
    Find minimum length subarray with sum >= target.
    Time: O(n), Space: O(1)
    """
    min_length = float('inf')
    window_sum = 0
    start = 0
    
    for end in range(len(arr)):
        window_sum += arr[end]
        
        # Shrink window while sum >= target
        while window_sum >= target:
            min_length = min(min_length, end - start + 1)
            window_sum -= arr[start]
            start += 1
    
    return min_length if min_length != float('inf') else 0

# Example
arr = [2, 3, 1, 2, 4, 3]
target = 7
result = min_subarray_sum_target(arr, target)
print(f"\nArray: {arr}, Target: {target}")
print(f"Minimum subarray length: {result}")


def substring_anagram(s: str, p: str) -> List[int]:
    """
    Find all starting indices of anagrams of p in s.
    Time: O(n), Space: O(1) - fixed alphabet
    """
    if len(p) > len(s):
        return []
    
    result = []
    p_count = Counter(p)
    window_count = Counter()
    
    # Initialize window
    for i in range(len(p)):
        window_count[s[i]] += 1
    
    if window_count == p_count:
        result.append(0)
    
    # Slide window
    for i in range(len(p), len(s)):
        # Add new character
        window_count[s[i]] += 1
        # Remove old character
        window_count[s[i - len(p)]] -= 1
        if window_count[s[i - len(p)]] == 0:
            del window_count[s[i - len(p)]]
        
        if window_count == p_count:
            result.append(i - len(p) + 1)
    
    return result

# Example
s = "cbaebabacd"
p = "abc"
result = substring_anagram(s, p)
print(f"\nString: '{s}', Pattern: '{p}'")
print(f"Anagram indices: {result}")


# ----------------------------------------------------------------------------
# 2.3 Sliding Window with Hash Map
# ----------------------------------------------------------------------------
print("\n--- 2.3 Sliding Window with Hash Map ---")

def longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Find longest substring with at most k distinct characters.
    Time: O(n), Space: O(k)
    """
    char_count = {}
    max_length = 0
    start = 0
    
    for end in range(len(s)):
        # Add character to window
        char_count[s[end]] = char_count.get(s[end], 0) + 1
        
        # Shrink window if more than k distinct
        while len(char_count) > k:
            char_count[s[start]] -= 1
            if char_count[s[start]] == 0:
                del char_count[s[start]]
            start += 1
        
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Example
s = "araaci"
k = 2
result = longest_substring_k_distinct(s, k)
print(f"String: '{s}', k: {k}")
print(f"Longest substring length: {result}")


# ============================================================================
# 3. PREFIX SUM
# ============================================================================

print("\n" + "=" * 70)
print("3. PREFIX SUM")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Basic Prefix Sum
# ----------------------------------------------------------------------------
print("\n--- 3.1 Basic Prefix Sum ---")

def prefix_sum(arr: List[int]) -> List[int]:
    """
    Calculate prefix sum array.
    prefix[i] = sum of arr[0] to arr[i]
    Time: O(n), Space: O(n)
    """
    prefix = [0] * len(arr)
    prefix[0] = arr[0]
    
    for i in range(1, len(arr)):
        prefix[i] = prefix[i - 1] + arr[i]
    
    return prefix

# Example
arr = [1, 2, 3, 4, 5]
prefix = prefix_sum(arr)
print(f"Array: {arr}")
print(f"Prefix sum: {prefix}")


def range_sum_query(prefix: List[int], left: int, right: int) -> int:
    """
    Get sum of range [left, right] using prefix sum.
    Time: O(1), Space: O(1)
    """
    if left == 0:
        return prefix[right]
    return prefix[right] - prefix[left - 1]

# Example
arr = [1, 2, 3, 4, 5]
prefix = prefix_sum(arr)
print(f"\nArray: {arr}")
print(f"Prefix: {prefix}")
print(f"Sum [1, 3]: {range_sum_query(prefix, 1, 3)}")
print(f"Sum [0, 4]: {range_sum_query(prefix, 0, 4)}")


# ----------------------------------------------------------------------------
# 3.2 Prefix Sum Applications
# ----------------------------------------------------------------------------
print("\n--- 3.2 Prefix Sum Applications ---")

def subarray_sum_equals_k(arr: List[int], k: int) -> int:
    """
    Count subarrays with sum equals k using prefix sum.
    Time: O(n), Space: O(n)
    """
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # prefix_sum: count
    
    for num in arr:
        prefix_sum += num
        
        # Check if prefix_sum - k exists
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]
        
        # Update prefix count
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1
    
    return count

# Example
arr = [1, 1, 1]
k = 2
result = subarray_sum_equals_k(arr, k)
print(f"Array: {arr}, k: {k}")
print(f"Number of subarrays with sum {k}: {result}")


def max_subarray_sum_kadane(arr: List[int]) -> int:
    """
    Maximum subarray sum using Kadane's algorithm (prefix sum variant).
    Time: O(n), Space: O(1)
    """
    max_sum = arr[0]
    current_sum = arr[0]
    
    for i in range(1, len(arr)):
        # Either extend previous subarray or start new
        current_sum = max(arr[i], current_sum + arr[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Example
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
result = max_subarray_sum_kadane(arr)
print(f"\nArray: {arr}")
print(f"Maximum subarray sum: {result}")


def product_except_self(arr: List[int]) -> List[int]:
    """
    Product of array except self using prefix and suffix products.
    Time: O(n), Space: O(1) excluding output
    """
    n = len(arr)
    result = [1] * n
    
    # Left products (prefix)
    for i in range(1, n):
        result[i] = result[i - 1] * arr[i - 1]
    
    # Right products (suffix)
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= arr[i]
    
    return result

# Example
arr = [1, 2, 3, 4]
result = product_except_self(arr)
print(f"\nArray: {arr}")
print(f"Product except self: {result}")


# ============================================================================
# 4. STRING PARSING
# ============================================================================

print("\n" + "=" * 70)
print("4. STRING PARSING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Basic String Parsing
# ----------------------------------------------------------------------------
print("\n--- 4.1 Basic String Parsing ---")

def parse_csv_line(line: str) -> List[str]:
    """
    Parse CSV line into list of values.
    Handles quoted fields.
    """
    result = []
    current_field = ""
    in_quotes = False
    
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            result.append(current_field)
            current_field = ""
        else:
            current_field += char
    
    result.append(current_field)
    return result

# Example
csv_line = 'Alice,25,"New York, NY",50000'
parsed = parse_csv_line(csv_line)
print(f"CSV line: {csv_line}")
print(f"Parsed: {parsed}")


def parse_key_value_pairs(s: str) -> dict:
    """
    Parse key=value pairs from string.
    Example: "name=Alice age=25 city=NY"
    """
    pairs = {}
    for pair in s.split():
        if '=' in pair:
            key, value = pair.split('=', 1)
            pairs[key] = value
    return pairs

# Example
kv_string = "name=Alice age=25 city=New York"
parsed = parse_key_value_pairs(kv_string)
print(f"\nKey-value string: {kv_string}")
print(f"Parsed: {parsed}")


# ----------------------------------------------------------------------------
# 4.2 Extracting Numbers from Strings
# ----------------------------------------------------------------------------
print("\n--- 4.2 Extracting Numbers from Strings ---")

def extract_numbers(s: str) -> List[int]:
    """
    Extract all integers from string.
    """
    import re
    return [int(x) for x in re.findall(r'-?\d+', s)]

# Example
text = "I have 5 apples and 3 oranges, total 8 fruits"
numbers = extract_numbers(text)
print(f"Text: '{text}'")
print(f"Numbers: {numbers}")


def extract_floats(s: str) -> List[float]:
    """
    Extract all floats from string.
    """
    import re
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', s)]

# Example
text = "Prices: $19.99, $29.50, and $5.00"
floats = extract_floats(text)
print(f"\nText: '{text}'")
print(f"Floats: {floats}")


# ----------------------------------------------------------------------------
# 4.3 Parsing Structured Data
# ----------------------------------------------------------------------------
print("\n--- 4.3 Parsing Structured Data ---")

def parse_email(email: str) -> dict:
    """
    Parse email into username and domain.
    """
    if '@' not in email:
        return None
    
    username, domain = email.split('@', 1)
    return {'username': username, 'domain': domain}

# Example
emails = ["alice@example.com", "bob.smith@company.co.uk"]
for email in emails:
    parsed = parse_email(email)
    print(f"Email: {email}, Parsed: {parsed}")


def parse_url(url: str) -> dict:
    """
    Basic URL parsing.
    """
    parts = {}
    
    # Protocol
    if '://' in url:
        protocol, rest = url.split('://', 1)
        parts['protocol'] = protocol
    else:
        rest = url
    
    # Domain and path
    if '/' in rest:
        domain, path = rest.split('/', 1)
        parts['domain'] = domain
        parts['path'] = '/' + path
    else:
        parts['domain'] = rest
    
    return parts

# Example
urls = ["https://www.example.com/path/to/page", "http://google.com"]
for url in urls:
    parsed = parse_url(url)
    print(f"\nURL: {url}")
    print(f"Parsed: {parsed}")


# ----------------------------------------------------------------------------
# 4.4 Parsing with Regular Expressions
# ----------------------------------------------------------------------------
print("\n--- 4.4 Parsing with Regular Expressions ---")

def parse_phone_number(phone: str) -> dict:
    """
    Parse phone number using regex.
    """
    import re
    pattern = r'(\d{3})-?(\d{3})-?(\d{4})'
    match = re.search(pattern, phone)
    
    if match:
        return {
            'area_code': match.group(1),
            'exchange': match.group(2),
            'number': match.group(3)
        }
    return None

# Example
phones = ["123-456-7890", "1234567890", "(123) 456-7890"]
for phone in phones:
    parsed = parse_phone_number(phone)
    print(f"Phone: {phone}, Parsed: {parsed}")


def parse_date(date_str: str) -> dict:
    """
    Parse date string (basic).
    """
    import re
    # Match YYYY-MM-DD or MM/DD/YYYY
    pattern1 = r'(\d{4})-(\d{2})-(\d{2})'
    pattern2 = r'(\d{2})/(\d{2})/(\d{4})'
    
    match = re.search(pattern1, date_str)
    if match:
        return {'year': match.group(1), 'month': match.group(2), 'day': match.group(3)}
    
    match = re.search(pattern2, date_str)
    if match:
        return {'year': match.group(3), 'month': match.group(1), 'day': match.group(2)}
    
    return None

# Example
dates = ["2023-12-25", "12/25/2023"]
for date in dates:
    parsed = parse_date(date)
    print(f"\nDate: {date}, Parsed: {parsed}")


# ----------------------------------------------------------------------------
# 4.5 Tokenization and Parsing
# ----------------------------------------------------------------------------
print("\n--- 4.5 Tokenization and Parsing ---")

def tokenize_expression(expr: str) -> List[str]:
    """
    Tokenize mathematical expression.
    """
    import re
    # Match numbers, operators, parentheses
    tokens = re.findall(r'\d+\.?\d*|[+\-*/()]', expr)
    return tokens

# Example
expressions = ["3+4*2", "10.5-2.3", "(1+2)*3"]
for expr in expressions:
    tokens = tokenize_expression(expr)
    print(f"Expression: {expr}, Tokens: {tokens}")


def parse_nested_brackets(s: str) -> List[str]:
    """
    Extract content from nested brackets.
    """
    result = []
    stack = []
    current = ""
    
    for char in s:
        if char == '(':
            if current:
                stack.append(current)
                current = ""
            stack.append('(')
        elif char == ')':
            if stack and stack[-1] == '(':
                if current:
                    result.append(current)
                    current = ""
                stack.pop()
                if stack:
                    current = stack.pop()
        else:
            current += char
    
    return result

# Example
text = "Hello (world (nested) here)"
parsed = parse_nested_brackets(text)
print(f"\nText: '{text}'")
print(f"Parsed: {parsed}")


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Two pointers - Three sum
print("\n--- Exercise 1: Three Sum ---")
def three_sum(arr: List[int], target: int) -> List[Tuple[int, int, int]]:
    """Find all triplets that sum to target."""
    arr.sort()
    result = []
    
    for i in range(len(arr) - 2):
        if i > 0 and arr[i] == arr[i - 1]:
            continue
        
        left, right = i + 1, len(arr) - 1
        while left < right:
            current_sum = arr[i] + arr[left] + arr[right]
            if current_sum == target:
                result.append((arr[i], arr[left], arr[right]))
                left += 1
                right -= 1
                while left < right and arr[left] == arr[left - 1]:
                    left += 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result

arr = [-1, 0, 1, 2, -1, -4]
target = 0
result = three_sum(arr, target)
print(f"Array: {arr}, Target: {target}")
print(f"Three sum: {result}")

# Exercise 2: Sliding window - Maximum average
print("\n--- Exercise 2: Maximum Average Subarray ---")
def max_average_subarray(arr: List[int], k: int) -> float:
    """Find maximum average of subarray of size k."""
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum / k

arr = [1, 12, -5, -6, 50, 3]
k = 4
result = max_average_subarray(arr, k)
print(f"Array: {arr}, k: {k}")
print(f"Maximum average: {result}")


# ============================================================================
# 6. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("6. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. TWO POINTERS:
   - Use for sorted arrays
   - One pointer at start, one at end
   - Move pointers based on condition
   - Time: O(n), Space: O(1)
   - Common: Two sum, palindromes, removing duplicates

2. SLIDING WINDOW:
   - Fixed size: Calculate first window, slide
   - Variable size: Expand/shrink based on condition
   - Use hash map for character counting
   - Time: O(n), Space: O(k) where k is window size
   - Common: Subarray sum, longest substring, anagrams

3. PREFIX SUM:
   - Precompute cumulative sums
   - Range queries in O(1)
   - Use hash map for subarray sum problems
   - Time: O(n) preprocessing, O(1) query
   - Common: Range sum, subarray sum equals k

4. STRING PARSING:
   - Use string methods for simple cases
   - Use regex for complex patterns
   - Consider state machines for nested structures
   - Handle edge cases (empty, invalid format)
   - Common: CSV parsing, extracting numbers, URL parsing

5. PATTERN RECOGNITION:
   - Two pointers: Sorted arrays, palindromes
   - Sliding window: Subarray/substring problems
   - Prefix sum: Range queries, cumulative problems
   - String parsing: Extract structured data

6. BEST PRACTICES:
   - Start with brute force, then optimize
   - Consider time/space complexity
   - Handle edge cases (empty, single element)
   - Use appropriate data structures
   - Test with examples
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Array/String Manipulation Guide Ready!")
    print("=" * 70)
