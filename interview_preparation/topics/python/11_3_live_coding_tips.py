"""
Python Live Coding Tips - Interview Preparation
Topic 11.3: Live Coding Tips

This module covers:
- Communication: Explain your thought process
- Start Simple: Get working solution first, optimize later
- Test Cases: Think about edge cases
- Code Cleanliness: Write readable code
"""

# ============================================================================
# 1. COMMUNICATION - EXPLAIN YOUR THOUGHT PROCESS
# ============================================================================

print("=" * 70)
print("1. COMMUNICATION - EXPLAIN YOUR THOUGHT PROCESS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Why Communication Matters
# ----------------------------------------------------------------------------
print("""
WHY COMMUNICATION MATTERS:
- Interviewers can't read your mind
- Shows problem-solving approach
- Demonstrates understanding
- Allows interviewer to guide you
- Shows collaboration skills
- More important than perfect code
""")


# ----------------------------------------------------------------------------
# 1.2 What to Communicate
# ----------------------------------------------------------------------------
print("\n--- What to Communicate ---")
print("""
1. UNDERSTANDING THE PROBLEM:
   "Let me make sure I understand..."
   "So we need to find..."
   "The input is... and output should be..."

2. YOUR APPROACH:
   "I'm thinking of using..."
   "This approach would be O(n²), let me think of better..."
   "I'll use a hash map because..."

3. AS YOU CODE:
   "I'm creating a dictionary to store..."
   "This loop will iterate through..."
   "I'm checking this condition because..."

4. WHEN STUCK:
   "I'm not sure about this part..."
   "Let me think about edge cases..."
   "Could I use a different data structure?"

5. TESTING:
   "Let me trace through with this example..."
   "For input [1,2,3], this should return..."
   "Edge case: empty list would..."
""")


# ----------------------------------------------------------------------------
# 1.3 Example: Good Communication
# ----------------------------------------------------------------------------
print("\n--- Example: Good Communication ---")
print("""
PROBLEM: Find two numbers that sum to target

GOOD COMMUNICATION:
"I understand we need to find two indices where the values sum to target.
Let me think about approaches:
1. Brute force: Check all pairs - O(n²) time
2. Hash map: Store complements - O(n) time

I'll go with the hash map approach for better efficiency.
I'll iterate through the array, for each number calculate the complement
(target - current), check if complement exists in hash map, if yes return
indices, otherwise add current number to hash map.

Let me code this step by step..."
""")


# ----------------------------------------------------------------------------
# 1.4 Example: Bad Communication
# ----------------------------------------------------------------------------
print("\n--- Example: Bad Communication ---")
print("""
BAD COMMUNICATION:
*Starts coding silently*
*Writes complex solution without explanation*
*Doesn't explain choices*
*Doesn't ask clarifying questions*

WHY IT'S BAD:
- Interviewer doesn't know your thought process
- Can't help if you're stuck
- Looks like you're guessing
- Doesn't show problem-solving skills
""")


# ============================================================================
# 2. START SIMPLE - GET WORKING SOLUTION FIRST
# ============================================================================

print("\n" + "=" * 70)
print("2. START SIMPLE - GET WORKING SOLUTION FIRST")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Why Start Simple
# ----------------------------------------------------------------------------
print("""
WHY START SIMPLE:
- Gets something working quickly
- Shows progress
- Easier to debug
- Can optimize later
- Better than perfect code that doesn't work
- Shows iterative improvement
""")


# ----------------------------------------------------------------------------
# 2.2 Example: Two Sum Problem
# ----------------------------------------------------------------------------
print("\n--- Example: Two Sum Problem ---")
print("""
PROBLEM: Find two numbers that sum to target

STEP 1: BRUTE FORCE (Start Simple)
""")

def two_sum_brute_force(nums, target):
    """
    Brute force approach - O(n²) time, O(1) space.
    Start with this to get working solution.
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

print("""
COMMUNICATION:
"I'll start with a brute force approach - check all pairs.
This is O(n²) but it's straightforward and will work.
Let me get this working first, then we can optimize."
""")

# Test
nums = [2, 7, 11, 15]
target = 9
result = two_sum_brute_force(nums, target)
print(f"Brute force result: {result}")

print("""
STEP 2: OPTIMIZE (After getting working solution)
""")

def two_sum_optimal(nums, target):
    """
    Optimal approach - O(n) time, O(n) space.
    Optimize after brute force works.
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

print("""
COMMUNICATION:
"Now that we have a working solution, let me optimize.
I can use a hash map to store numbers I've seen.
For each number, I check if its complement exists.
This reduces time complexity to O(n)."
""")

result = two_sum_optimal(nums, target)
print(f"Optimal result: {result}")


# ----------------------------------------------------------------------------
# 2.3 Example: Finding Maximum
# ----------------------------------------------------------------------------
print("\n--- Example: Finding Maximum ---")
print("""
STEP 1: SIMPLE APPROACH
""")

def find_max_simple(numbers):
    """Simple approach - use built-in."""
    if not numbers:
        return None
    return max(numbers)

print("""
COMMUNICATION:
"I'll start with the built-in max() function.
This is simple and works. If needed, I can implement
it manually to show the algorithm."
""")

numbers = [3, 7, 2, 9, 1, 5]
print(f"Max (simple): {find_max_simple(numbers)}")

print("""
STEP 2: MANUAL IMPLEMENTATION (If asked)
""")

def find_max_manual(numbers):
    """Manual implementation - shows algorithm understanding."""
    if not numbers:
        return None
    
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val

print("""
COMMUNICATION:
"If you'd like, I can implement it manually.
I'll iterate through and track the maximum.
This shows the O(n) algorithm."
""")

print(f"Max (manual): {find_max_manual(numbers)}")


# ============================================================================
# 3. TEST CASES - THINK ABOUT EDGE CASES
# ============================================================================

print("\n" + "=" * 70)
print("3. TEST CASES - THINK ABOUT EDGE CASES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Common Edge Cases
# ----------------------------------------------------------------------------
print("""
COMMON EDGE CASES TO CONSIDER:

1. EMPTY INPUTS:
   - Empty list: []
   - Empty string: ""
   - None/null values

2. SINGLE ELEMENT:
   - List with one item: [5]
   - String with one char: "a"

3. BOUNDARY VALUES:
   - Minimum: 0, negative numbers
   - Maximum: large numbers, overflow
   - At boundaries: first/last element

4. DUPLICATES:
   - All same values: [1, 1, 1]
   - Some duplicates: [1, 2, 2, 3]

5. SPECIAL CASES:
   - Already sorted/unsorted
   - All positive/negative
   - Mixed types (if applicable)
""")


# ----------------------------------------------------------------------------
# 3.2 Example: Testing Function
# ----------------------------------------------------------------------------
print("\n--- Example: Testing Function ---")

def find_max_with_tests(numbers):
    """Find maximum with edge case handling."""
    # Edge case: Empty input
    if not numbers:
        return None
    
    # Edge case: Single element
    if len(numbers) == 1:
        return numbers[0]
    
    # Normal case
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    
    return max_val

print("""
TEST CASES TO MENTION:
1. Empty list: [] → None
2. Single element: [5] → 5
3. All same: [3, 3, 3] → 3
4. Negative numbers: [-1, -5, -3] → -1
5. Mixed: [1, -2, 5, 0] → 5
6. Large numbers: [1000, 999, 1001] → 1001
""")

# Test cases
test_cases = [
    ([], None),
    ([5], 5),
    ([3, 3, 3], 3),
    ([-1, -5, -3], -1),
    ([1, -2, 5, 0], 5),
    ([1000, 999, 1001], 1001)
]

print("Running test cases:")
for nums, expected in test_cases:
    result = find_max_with_tests(nums)
    status = "✓" if result == expected else "✗"
    print(f"{status} Input: {nums}, Expected: {expected}, Got: {result}")


# ----------------------------------------------------------------------------
# 3.3 Example: String Processing with Edge Cases
# ----------------------------------------------------------------------------
print("\n--- Example: String Processing with Edge Cases ---")

def reverse_string(text):
    """Reverse string with edge case handling."""
    # Edge cases
    if not text:  # Empty string
        return ""
    
    if len(text) == 1:  # Single character
        return text
    
    # Normal case
    return text[::-1]

print("""
EDGE CASES FOR STRING REVERSAL:
1. Empty string: "" → ""
2. Single char: "a" → "a"
3. Palindrome: "aba" → "aba"
4. Special chars: "!@#" → "#@!"
5. Whitespace: "  " → "  "
""")

test_cases = [
    ("", ""),
    ("a", "a"),
    ("hello", "olleh"),
    ("aba", "aba"),
    ("!@#", "#@!"),
    ("  ", "  ")
]

print("Testing string reversal:")
for text, expected in test_cases:
    result = reverse_string(text)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{text}' → '{result}' (expected '{expected}')")


# ----------------------------------------------------------------------------
# 3.4 How to Present Test Cases
# ----------------------------------------------------------------------------
print("\n--- How to Present Test Cases ---")
print("""
WHEN PRESENTING TEST CASES:

1. VERBALIZE YOUR THINKING:
   "Let me think about edge cases:
   - Empty input: should return None
   - Single element: should return that element
   - All duplicates: should still work
   - Negative numbers: should handle correctly"

2. WALK THROUGH EXAMPLES:
   "For input [2, 7, 11, 15] and target 9:
   - Check 2 + 7 = 9 ✓
   - Return [0, 1]"

3. TEST AS YOU CODE:
   "Let me test this with an empty list..."
   "What if all numbers are the same?"

4. MENTION EDGE CASES EVEN IF NOT TESTING:
   "This should handle empty inputs by returning None"
   "Single element case is covered by the loop"
""")


# ============================================================================
# 4. CODE CLEANLINESS - WRITE READABLE CODE
# ============================================================================

print("\n" + "=" * 70)
print("4. CODE CLEANLINESS - WRITE READABLE CODE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Clean Code Principles
# ----------------------------------------------------------------------------
print("""
CLEAN CODE PRINCIPLES:

1. MEANINGFUL NAMES:
   ✓ total_price, user_count, is_valid
   ✗ tp, uc, v

2. SMALL FUNCTIONS:
   ✓ Single responsibility
   ✗ One function does everything

3. COMMENTS:
   ✓ Explain WHY, not WHAT
   ✗ Obvious comments

4. FORMATTING:
   ✓ Consistent indentation
   ✓ Proper spacing
   ✗ Messy, inconsistent

5. STRUCTURE:
   ✓ Logical flow
   ✓ Clear organization
   ✗ Jumping around
""")


# ----------------------------------------------------------------------------
# 4.2 Example: Clean vs Messy Code
# ----------------------------------------------------------------------------
print("\n--- Example: Clean vs Messy Code ---")
print("""
MESSY CODE:
""")

# Messy version
def f(lst, t):
    d = {}
    for i in range(len(lst)):
        c = t - lst[i]
        if c in d:
            return [d[c], i]
        d[lst[i]] = i
    return []

print("""
PROBLEMS:
- Unclear variable names (f, lst, t, d, c)
- No docstring
- No comments
- Hard to understand
""")

print("""
CLEAN CODE:
""")

# Clean version
def two_sum(numbers, target):
    """
    Find two indices where values sum to target.
    
    Args:
        numbers: List of integers
        target: Target sum
    
    Returns:
        List of two indices, or empty list if not found
    """
    # Map: number -> index
    number_to_index = {}
    
    for current_index, current_number in enumerate(numbers):
        # Calculate complement needed
        complement = target - current_number
        
        # Check if complement exists
        if complement in number_to_index:
            return [number_to_index[complement], current_index]
        
        # Store current number for future lookups
        number_to_index[current_number] = current_index
    
    return []

print("""
IMPROVEMENTS:
✓ Descriptive function name
✓ Clear variable names
✓ Docstring explaining purpose
✓ Comments explaining logic
✓ Easy to understand
""")


# ----------------------------------------------------------------------------
# 4.3 Code Organization
# ----------------------------------------------------------------------------
print("\n--- Code Organization ---")
print("""
GOOD CODE ORGANIZATION:

1. FUNCTION STRUCTURE:
   - Docstring
   - Edge case handling
   - Main logic
   - Return statement

2. VARIABLE NAMING:
   - Descriptive names
   - Consistent style
   - Avoid abbreviations

3. LOGICAL FLOW:
   - Top to bottom
   - Clear steps
   - Group related operations

4. SPACING:
   - Blank lines between logical sections
   - Consistent indentation
   - Proper alignment
""")


# ----------------------------------------------------------------------------
# 4.4 Example: Well-Organized Function
# ----------------------------------------------------------------------------
print("\n--- Example: Well-Organized Function ---")

def find_duplicates(numbers):
    """
    Find all duplicate numbers in list.
    
    Args:
        numbers: List of integers
    
    Returns:
        List of duplicate numbers (preserving order)
    """
    # Edge case: Empty or single element
    if len(numbers) <= 1:
        return []
    
    # Track seen numbers and duplicates
    seen = set()
    duplicates = []
    
    # Find duplicates
    for number in numbers:
        if number in seen:
            # Add to duplicates if not already added
            if number not in duplicates:
                duplicates.append(number)
        else:
            seen.add(number)
    
    return duplicates

print("""
ORGANIZATION:
✓ Clear docstring
✓ Edge case handling first
✓ Descriptive variable names
✓ Comments for each section
✓ Logical flow
✓ Easy to read and understand
""")

# Test
numbers = [1, 2, 3, 2, 4, 3, 5, 2]
result = find_duplicates(numbers)
print(f"Duplicates: {result}")


# ============================================================================
# 5. COMPLETE EXAMPLE: SOLVING A PROBLEM
# ============================================================================

print("\n" + "=" * 70)
print("5. COMPLETE EXAMPLE: SOLVING A PROBLEM")
print("=" * 70)

print("""
PROBLEM: Check if string is palindrome

STEP-BY-STEP APPROACH:
""")

print("""
STEP 1: UNDERSTAND & COMMUNICATE
"I need to check if a string reads the same forwards and backwards.
Examples: 'racecar' is palindrome, 'hello' is not.
I should handle case sensitivity - should 'Racecar' be considered?
Let me assume case-sensitive for now."
""")

print("""
STEP 2: START SIMPLE
"I'll start with a simple approach: compare string with its reverse.
This is straightforward and will work."
""")

def is_palindrome_simple(text):
    """Simple palindrome check."""
    if not text:
        return True  # Empty string is palindrome
    
    return text == text[::-1]

print("""
STEP 3: TEST WITH EXAMPLES
"Let me test:
- 'racecar' == 'racecar'[::-1] → True ✓
- 'hello' == 'hello'[::-1] → False ✓
- '' == ''[::-1] → True ✓"
""")

test_cases = [
    ("racecar", True),
    ("hello", False),
    ("", True),
    ("a", True),
    ("ab", False)
]

print("Testing simple version:")
for text, expected in test_cases:
    result = is_palindrome_simple(text)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{text}': {result}")

print("""
STEP 4: HANDLE EDGE CASES
"Edge cases:
- Empty string: handled ✓
- Single character: handled ✓
- Case sensitivity: currently case-sensitive
- Whitespace: 'a a' would be False, might want to handle"
""")

print("""
STEP 5: OPTIMIZE IF NEEDED
"If interviewer wants case-insensitive or ignore spaces,
I can modify the function. For now, simple version works."
""")

def is_palindrome_advanced(text):
    """Advanced: case-insensitive, ignore spaces."""
    # Normalize: lowercase and remove spaces
    normalized = ''.join(c.lower() for c in text if c.isalnum())
    
    if not normalized:
        return True
    
    return normalized == normalized[::-1]

print("""
STEP 6: CLEAN CODE
"Function is clean:
✓ Descriptive name
✓ Handles edge cases
✓ Clear logic
✓ Easy to understand"
""")


# ============================================================================
# 6. COMMON MISTAKES TO AVOID
# ============================================================================

print("\n" + "=" * 70)
print("6. COMMON MISTAKES TO AVOID")
print("=" * 70)

print("""
MISTAKES TO AVOID:

1. JUMPING TO CODE TOO QUICKLY:
   ✗ Start coding immediately
   ✓ Understand problem first, ask questions

2. NOT EXPLAINING:
   ✗ Code silently
   ✓ Explain your thought process

3. PERFECTIONISM:
   ✗ Try to write perfect code first
   ✓ Get working solution, optimize later

4. IGNORING EDGE CASES:
   ✗ Only test happy path
   ✓ Think about edge cases

5. MESSY CODE:
   ✗ Short variable names, no structure
   ✓ Clean, readable code

6. NOT TESTING:
   ✗ Assume code works
   ✓ Test with examples

7. GIVING UP TOO QUICKLY:
   ✗ Say "I don't know" immediately
   ✓ Think out loud, ask for hints

8. NOT ASKING QUESTIONS:
   ✗ Assume requirements
   ✓ Clarify constraints and requirements
""")


# ============================================================================
# 7. INTERVIEW WORKFLOW
# ============================================================================

print("\n" + "=" * 70)
print("7. INTERVIEW WORKFLOW")
print("=" * 70)

print("""
RECOMMENDED WORKFLOW:

1. READ PROBLEM CAREFULLY (2-3 min)
   - Understand requirements
   - Identify inputs/outputs
   - Note constraints

2. ASK CLARIFYING QUESTIONS (1-2 min)
   - Edge cases?
   - Input format?
   - Expected output format?
   - Performance requirements?

3. THINK OUT LOUD (2-3 min)
   - Discuss approaches
   - Mention trade-offs
   - Choose approach

4. START SIMPLE (5-10 min)
   - Get working solution
   - Explain as you code
   - Use meaningful names

5. TEST YOUR CODE (2-3 min)
   - Walk through examples
   - Test edge cases
   - Verify correctness

6. OPTIMIZE IF TIME (5-10 min)
   - Discuss improvements
   - Implement if time allows
   - Explain trade-offs

7. DISCUSS (2-3 min)
   - Time/space complexity
   - Alternative approaches
   - Trade-offs
""")


# ============================================================================
# 8. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("8. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. COMMUNICATION:
   - Explain your thought process
   - Think out loud
   - Ask clarifying questions
   - Discuss trade-offs

2. START SIMPLE:
   - Get working solution first
   - Optimize later if time
   - Better working code than perfect broken code

3. TEST CASES:
   - Think about edge cases
   - Test as you code
   - Walk through examples
   - Verify correctness

4. CODE CLEANLINESS:
   - Meaningful variable names
   - Clear structure
   - Comments for complex logic
   - Readable code

5. MINDSET:
   - It's a conversation, not a test
   - Interviewer wants you to succeed
   - Show problem-solving process
   - Don't panic if stuck

6. PRACTICE:
   - Practice explaining out loud
   - Practice starting simple
   - Practice thinking about edge cases
   - Practice writing clean code

7. TIME MANAGEMENT:
   - Don't spend too long on one approach
   - Get something working quickly
   - Optimize if time permits
   - Communicate time constraints
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Live Coding Tips Guide Ready!")
    print("=" * 70)
    print("\nRemember:")
    print("- Communicate your thought process")
    print("- Start simple, optimize later")
    print("- Think about edge cases")
    print("- Write clean, readable code")
    print("\nGood luck with your interview!")
