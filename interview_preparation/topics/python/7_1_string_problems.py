"""
Python String Problems - Interview Preparation
Topic 7.1: String Problems

This module covers:
- Anagram Detection: Checking if strings are anagrams
- Palindrome Checking: Reversing strings, checking palindromes
- String Matching: Finding substrings, pattern matching
- String Transformation: Replacing, splitting, joining
"""

from collections import Counter, defaultdict
import re

# ============================================================================
# 1. ANAGRAM DETECTION
# ============================================================================

print("=" * 70)
print("1. ANAGRAM DETECTION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Basic Anagram Check
# ----------------------------------------------------------------------------
print("\n--- 1.1 Basic Anagram Check ---")

def is_anagram_sorted(s1, s2):
    """
    Check if two strings are anagrams using sorting.
    Time: O(n log n), Space: O(n)
    """
    # Remove spaces and convert to lowercase
    s1_clean = ''.join(s1.lower().split())
    s2_clean = ''.join(s2.lower().split())
    
    # Check if sorted characters are equal
    return sorted(s1_clean) == sorted(s2_clean)

# Test cases
test_cases = [
    ("listen", "silent", True),
    ("racecar", "carrace", True),
    ("hello", "world", False),
    ("", "", True),
    ("a", "a", True),
    ("a", "b", False)
]

print("Testing anagram detection:")
for s1, s2, expected in test_cases:
    result = is_anagram_sorted(s1, s2)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{s1}' & '{s2}': {result} (expected {expected})")


# ----------------------------------------------------------------------------
# 1.2 Anagram Check with Counter
# ----------------------------------------------------------------------------
print("\n--- 1.2 Anagram Check with Counter ---")

def is_anagram_counter(s1, s2):
    """
    Check if two strings are anagrams using Counter.
    Time: O(n), Space: O(n)
    """
    s1_clean = ''.join(s1.lower().split())
    s2_clean = ''.join(s2.lower().split())
    
    if len(s1_clean) != len(s2_clean):
        return False
    
    return Counter(s1_clean) == Counter(s2_clean)

print("Testing Counter approach:")
for s1, s2, expected in test_cases:
    result = is_anagram_counter(s1, s2)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{s1}' & '{s2}': {result}")


# ----------------------------------------------------------------------------
# 1.3 Anagram Check with Frequency Array
# ----------------------------------------------------------------------------
print("\n--- 1.3 Anagram Check with Frequency Array ---")

def is_anagram_frequency(s1, s2):
    """
    Check if two strings are anagrams using frequency array.
    Assumes lowercase English letters only.
    Time: O(n), Space: O(1) - fixed 26 characters
    """
    s1_clean = s1.lower().replace(' ', '')
    s2_clean = s2.lower().replace(' ', '')
    
    if len(s1_clean) != len(s2_clean):
        return False
    
    # Frequency array for 26 letters
    freq = [0] * 26
    
    # Count characters in s1
    for char in s1_clean:
        freq[ord(char) - ord('a')] += 1
    
    # Subtract characters in s2
    for char in s2_clean:
        index = ord(char) - ord('a')
        freq[index] -= 1
        if freq[index] < 0:
            return False
    
    # All frequencies should be 0
    return all(f == 0 for f in freq)

print("Testing frequency array approach:")
for s1, s2, expected in test_cases[:4]:  # Test first 4
    result = is_anagram_frequency(s1, s2)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{s1}' & '{s2}': {result}")


# ----------------------------------------------------------------------------
# 1.4 Group Anagrams
# ----------------------------------------------------------------------------
print("\n--- 1.4 Group Anagrams ---")

def group_anagrams(words):
    """
    Group words that are anagrams of each other.
    Time: O(n * k log k) where k is average word length
    Space: O(n * k)
    """
    groups = defaultdict(list)
    
    for word in words:
        # Create key from sorted characters
        key = ''.join(sorted(word.lower()))
        groups[key].append(word)
    
    return list(groups.values())

words = ["eat", "tea", "tan", "ate", "nat", "bat"]
grouped = group_anagrams(words)
print(f"Grouped anagrams: {grouped}")


# ============================================================================
# 2. PALINDROME CHECKING
# ============================================================================

print("\n" + "=" * 70)
print("2. PALINDROME CHECKING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic Palindrome Check
# ----------------------------------------------------------------------------
print("\n--- 2.1 Basic Palindrome Check ---")

def is_palindrome_simple(text):
    """
    Check if string is palindrome using slicing.
    Time: O(n), Space: O(n) for reversed string
    """
    if not text:
        return True
    
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]

test_cases = [
    ("racecar", True),
    ("hello", False),
    ("", True),
    ("a", True),
    ("A man a plan a canal Panama", True),  # With spaces
    ("race a car", False)
]

print("Testing palindrome detection:")
for text, expected in test_cases:
    result = is_palindrome_simple(text)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{text}': {result} (expected {expected})")


# ----------------------------------------------------------------------------
# 2.2 Palindrome Check with Two Pointers
# ----------------------------------------------------------------------------
print("\n--- 2.2 Palindrome Check with Two Pointers ---")

def is_palindrome_two_pointers(text):
    """
    Check palindrome using two pointers.
    Time: O(n), Space: O(1)
    """
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    
    if not cleaned:
        return True
    
    left, right = 0, len(cleaned) - 1
    
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    
    return True

print("Testing two pointers approach:")
for text, expected in test_cases:
    result = is_palindrome_two_pointers(text)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{text}': {result}")


# ----------------------------------------------------------------------------
# 2.3 Reverse String
# ----------------------------------------------------------------------------
print("\n--- 2.3 Reverse String ---")

def reverse_string_slicing(text):
    """Reverse string using slicing - most Pythonic."""
    return text[::-1]

def reverse_string_builtin(text):
    """Reverse string using built-in reversed()."""
    return ''.join(reversed(text))

def reverse_string_loop(text):
    """Reverse string using loop."""
    result = []
    for i in range(len(text) - 1, -1, -1):
        result.append(text[i])
    return ''.join(result)

text = "hello"
print(f"Original: '{text}'")
print(f"Slicing: '{reverse_string_slicing(text)}'")
print(f"Built-in: '{reverse_string_builtin(text)}'")
print(f"Loop: '{reverse_string_loop(text)}'")


# ----------------------------------------------------------------------------
# 2.4 Longest Palindromic Substring
# ----------------------------------------------------------------------------
print("\n--- 2.4 Longest Palindromic Substring ---")

def longest_palindrome_expand(s):
    """
    Find longest palindromic substring using expand around center.
    Time: O(n²), Space: O(1)
    """
    if not s:
        return ""
    
    def expand_around_center(left, right):
        """Expand around center and return palindrome length."""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = ""
    
    for i in range(len(s)):
        # Odd length palindrome
        pal1 = expand_around_center(i, i)
        if len(pal1) > len(longest):
            longest = pal1
        
        # Even length palindrome
        pal2 = expand_around_center(i, i + 1)
        if len(pal2) > len(longest):
            longest = pal2
    
    return longest

test_strings = ["babad", "cbbd", "racecar"]
for s in test_strings:
    result = longest_palindrome_expand(s)
    print(f"'{s}' → longest palindrome: '{result}'")


# ============================================================================
# 3. STRING MATCHING
# ============================================================================

print("\n" + "=" * 70)
print("3. STRING MATCHING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Find Substring
# ----------------------------------------------------------------------------
print("\n--- 3.1 Find Substring ---")

def find_substring(text, pattern):
    """
    Find all occurrences of pattern in text.
    Time: O(n * m) where n=text length, m=pattern length
    """
    indices = []
    pattern_len = len(pattern)
    
    for i in range(len(text) - pattern_len + 1):
        if text[i:i + pattern_len] == pattern:
            indices.append(i)
    
    return indices

text = "hello world hello"
pattern = "hello"
indices = find_substring(text, pattern)
print(f"Pattern '{pattern}' found at indices: {indices}")


# ----------------------------------------------------------------------------
# 3.2 Pattern Matching with Regex
# ----------------------------------------------------------------------------
print("\n--- 3.2 Pattern Matching with Regex ---")

def find_pattern_regex(text, pattern):
    """Find pattern using regular expressions."""
    matches = re.findall(pattern, text)
    match_indices = [m.start() for m in re.finditer(pattern, text)]
    return matches, match_indices

text = "The price is $100 and $200"
pattern = r'\$\d+'
matches, indices = find_pattern_regex(text, pattern)
print(f"Pattern '{pattern}' found: {matches} at {indices}")


# ----------------------------------------------------------------------------
# 3.3 Check if String Contains Pattern
# ----------------------------------------------------------------------------
print("\n--- 3.3 Check if String Contains Pattern ---")

def contains_pattern(text, pattern):
    """Check if text contains pattern."""
    return pattern in text

def starts_with_pattern(text, pattern):
    """Check if text starts with pattern."""
    return text.startswith(pattern)

def ends_with_pattern(text, pattern):
    """Check if text ends with pattern."""
    return text.endswith(pattern)

text = "hello world"
print(f"Contains 'world': {contains_pattern(text, 'world')}")
print(f"Starts with 'hello': {starts_with_pattern(text, 'hello')}")
print(f"Ends with 'world': {ends_with_pattern(text, 'world')}")


# ----------------------------------------------------------------------------
# 3.4 Longest Common Substring
# ----------------------------------------------------------------------------
print("\n--- 3.4 Longest Common Substring ---")

def longest_common_substring(s1, s2):
    """
    Find longest common substring between two strings.
    Time: O(n * m), Space: O(n * m)
    """
    if not s1 or not s2:
        return ""
    
    m, n = len(s1), len(s2)
    # dp[i][j] = length of common substring ending at s1[i] and s2[j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_pos = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i
    
    return s1[end_pos - max_len:end_pos] if max_len > 0 else ""

s1, s2 = "abcdef", "abcef"
result = longest_common_substring(s1, s2)
print(f"Longest common substring of '{s1}' and '{s2}': '{result}'")


# ============================================================================
# 4. STRING TRANSFORMATION
# ============================================================================

print("\n" + "=" * 70)
print("4. STRING TRANSFORMATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Replacing Substrings
# ----------------------------------------------------------------------------
print("\n--- 4.1 Replacing Substrings ---")

def replace_substring(text, old, new):
    """Replace all occurrences of old with new."""
    return text.replace(old, new)

text = "hello world hello"
result = replace_substring(text, "hello", "hi")
print(f"Replace 'hello' with 'hi': '{result}'")

# Replace with count limit
result = text.replace("hello", "hi", 1)  # Replace only first occurrence
print(f"Replace first occurrence: '{result}'")


# ----------------------------------------------------------------------------
# 4.2 Splitting Strings
# ----------------------------------------------------------------------------
print("\n--- 4.2 Splitting Strings ---")

def split_examples():
    """Demonstrate string splitting."""
    text = "apple,banana,cherry"
    
    # Split by comma
    parts = text.split(',')
    print(f"Split by comma: {parts}")
    
    # Split with max splits
    parts = text.split(',', 1)  # Split into 2 parts max
    print(f"Split with max: {parts}")
    
    # Split by whitespace
    text2 = "hello   world   python"
    parts = text2.split()  # Splits on any whitespace
    print(f"Split whitespace: {parts}")
    
    # Split lines
    text3 = "line1\nline2\nline3"
    lines = text3.splitlines()
    print(f"Split lines: {lines}")

split_examples()


# ----------------------------------------------------------------------------
# 4.3 Joining Strings
# ----------------------------------------------------------------------------
print("\n--- 4.3 Joining Strings ---")

def join_examples():
    """Demonstrate string joining."""
    words = ["hello", "world", "python"]
    
    # Join with space
    result = ' '.join(words)
    print(f"Join with space: '{result}'")
    
    # Join with comma
    result = ','.join(words)
    print(f"Join with comma: '{result}'")
    
    # Join with no separator
    result = ''.join(words)
    print(f"Join no separator: '{result}'")
    
    # Join characters
    chars = ['h', 'e', 'l', 'l', 'o']
    result = ''.join(chars)
    print(f"Join characters: '{result}'")

join_examples()


# ----------------------------------------------------------------------------
# 4.4 String Case Transformations
# ----------------------------------------------------------------------------
print("\n--- 4.4 String Case Transformations ---")

def case_transformations():
    """Demonstrate case transformations."""
    text = "Hello World Python"
    
    print(f"Original: '{text}'")
    print(f"Lower: '{text.lower()}'")
    print(f"Upper: '{text.upper()}'")
    print(f"Title: '{text.title()}'")
    print(f"Capitalize: '{text.capitalize()}'")
    print(f"Swapcase: '{text.swapcase()}'")

case_transformations()


# ----------------------------------------------------------------------------
# 4.5 Remove Characters
# ----------------------------------------------------------------------------
print("\n--- 4.5 Remove Characters ---")

def remove_characters(text, chars_to_remove):
    """Remove specified characters from string."""
    return ''.join(c for c in text if c not in chars_to_remove)

def remove_whitespace(text):
    """Remove all whitespace."""
    return ''.join(text.split())

def remove_punctuation(text):
    """Remove punctuation."""
    return ''.join(c for c in text if c.isalnum() or c.isspace())

text = "Hello, World! Python 3.9"
print(f"Original: '{text}'")
print(f"Remove punctuation: '{remove_punctuation(text)}'")
print(f"Remove whitespace: '{remove_whitespace(text)}'")


# ----------------------------------------------------------------------------
# 4.6 String Padding
# ----------------------------------------------------------------------------
print("\n--- 4.6 String Padding ---")

def padding_examples():
    """Demonstrate string padding."""
    text = "hello"
    
    print(f"Original: '{text}'")
    print(f"Left pad (10): '{text.ljust(10)}'")
    print(f"Right pad (10): '{text.rjust(10)}'")
    print(f"Center pad (10): '{text.center(10)}'")
    print(f"Left pad with zeros: '{text.zfill(10)}'")
    print(f"Right pad with '*': '{text.ljust(10, '*')}'")

padding_examples()


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Valid Anagram
print("\n--- Exercise 1: Valid Anagram ---")
def valid_anagram(s, t):
    """Check if s and t are anagrams."""
    if len(s) != len(t):
        return False
    return Counter(s.lower()) == Counter(t.lower())

print(f"valid_anagram('listen', 'silent'): {valid_anagram('listen', 'silent')}")
print(f"valid_anagram('rat', 'car'): {valid_anagram('rat', 'car')}")


# Exercise 2: Valid Palindrome
print("\n--- Exercise 2: Valid Palindrome ---")
def valid_palindrome(s):
    """Check if string is palindrome (alphanumeric only)."""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

print(f"valid_palindrome('A man a plan a canal Panama'): {valid_palindrome('A man a plan a canal Panama')}")
print(f"valid_palindrome('race a car'): {valid_palindrome('race a car')}")


# Exercise 3: First Unique Character
print("\n--- Exercise 3: First Unique Character ---")
def first_unique_char(s):
    """Find index of first unique character."""
    freq = Counter(s)
    for i, char in enumerate(s):
        if freq[char] == 1:
            return i
    return -1

print(f"first_unique_char('leetcode'): {first_unique_char('leetcode')}")
print(f"first_unique_char('loveleetcode'): {first_unique_char('loveleetcode')}")


# Exercise 4: Reverse Words in String
print("\n--- Exercise 4: Reverse Words in String ---")
def reverse_words(s):
    """Reverse order of words in string."""
    words = s.split()
    return ' '.join(reversed(words))

print(f"reverse_words('the sky is blue'): '{reverse_words('the sky is blue')}'")


# Exercise 5: Longest Common Prefix
print("\n--- Exercise 5: Longest Common Prefix ---")
def longest_common_prefix(strs):
    """Find longest common prefix among strings."""
    if not strs:
        return ""
    
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

words = ["flower", "flow", "flight"]
print(f"longest_common_prefix({words}): '{longest_common_prefix(words)}'")


# ============================================================================
# 6. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("6. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. ANAGRAM DETECTION:
   - Sort and compare: O(n log n)
   - Counter: O(n) - most Pythonic
   - Frequency array: O(n), O(1) space for fixed alphabet
   - Check length first for early exit

2. PALINDROME CHECKING:
   - Slicing: text == text[::-1] - simplest
   - Two pointers: O(n) time, O(1) space
   - Clean string first (remove spaces, punctuation)
   - Handle case sensitivity

3. STRING MATCHING:
   - Use 'in' operator for simple checks
   - Use regex for complex patterns
   - Two pointers for substring problems
   - Dynamic programming for longest common substring

4. STRING TRANSFORMATION:
   - Use string methods when possible
   - join() is faster than + for multiple strings
   - split() and join() are powerful
   - Remember strings are immutable

5. COMMON PATTERNS:
   - Counter for frequency analysis
   - Two pointers for palindromes/substrings
   - Sliding window for substring problems
   - Regex for pattern matching

6. BEST PRACTICES:
   - Handle empty strings
   - Consider case sensitivity
   - Remove spaces/punctuation when needed
   - Use appropriate data structures
   - Consider time/space complexity
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("String Problems Guide Ready!")
    print("=" * 70)
