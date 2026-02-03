"""
Dynamic Programming (Basic) - Interview Preparation
Topic 6.5: Dynamic Programming (Basic)

This module covers:
- Memoization: Caching results
- Tabulation: Bottom-up approach
- Common Patterns: Fibonacci, coin change, longest common subsequence (basic)
"""

from functools import lru_cache
from typing import Dict, List

# ============================================================================
# 1. UNDERSTANDING DYNAMIC PROGRAMMING
# ============================================================================

print("=" * 70)
print("1. UNDERSTANDING DYNAMIC PROGRAMMING")
print("=" * 70)

print("""
DYNAMIC PROGRAMMING APPROACH:
- Solves problems by breaking into overlapping subproblems
- Stores results of subproblems to avoid recomputation
- Two main approaches: Memoization (top-down) and Tabulation (bottom-up)

KEY CHARACTERISTICS:
1. Overlapping Subproblems: Same subproblems appear multiple times
2. Optimal Substructure: Optimal solution contains optimal subproblem solutions
3. Memoization/Tabulation: Store results to avoid recomputation

WHEN TO USE DP:
- Problem has overlapping subproblems
- Optimal substructure exists
- Recursive solution has repeated calculations
- Need to optimize recursive solution

TWO APPROACHES:
1. Memoization (Top-Down): Recursive + caching
2. Tabulation (Bottom-Up): Iterative + table filling
""")


# ============================================================================
# 2. MEMOIZATION (TOP-DOWN)
# ============================================================================

print("\n" + "=" * 70)
print("2. MEMOIZATION (TOP-DOWN)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Manual Memoization
# ----------------------------------------------------------------------------
print("\n--- 2.1 Manual Memoization ---")

def fibonacci_recursive(n: int) -> int:
    """
    Fibonacci without memoization - inefficient.
    Time: O(2^n), Space: O(n) for call stack
    """
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# This is too slow for large n!


def fibonacci_memoized(n: int, memo: Dict[int, int] = None) -> int:
    """
    Fibonacci with manual memoization.
    Time: O(n), Space: O(n)
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n < 2:
        result = n
    else:
        result = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    
    memo[n] = result
    return result

# Example
result = fibonacci_memoized(10)
print(f"Fibonacci(10) with memoization: {result}")


# ----------------------------------------------------------------------------
# 2.2 Memoization Decorator
# ----------------------------------------------------------------------------
print("\n--- 2.2 Memoization Decorator ---")

def memoize(func):
    """Generic memoization decorator."""
    cache = {}
    
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    wrapper.cache = cache
    return wrapper

@memoize
def fibonacci_decorated(n: int) -> int:
    """Fibonacci with decorator memoization."""
    if n < 2:
        return n
    return fibonacci_decorated(n - 1) + fibonacci_decorated(n - 2)

# Example
result = fibonacci_decorated(10)
print(f"Fibonacci(10) with decorator: {result}")


# ----------------------------------------------------------------------------
# 2.3 Using functools.lru_cache
# ----------------------------------------------------------------------------
print("\n--- 2.3 Using functools.lru_cache ---")

@lru_cache(maxsize=128)
def fibonacci_lru(n: int) -> int:
    """Fibonacci with lru_cache."""
    if n < 2:
        return n
    return fibonacci_lru(n - 1) + fibonacci_lru(n - 2)

# Example
result = fibonacci_lru(10)
print(f"Fibonacci(10) with lru_cache: {result}")
print(f"Cache info: {fibonacci_lru.cache_info()}")


# ----------------------------------------------------------------------------
# 2.4 Memoization Example - Factorial
# ----------------------------------------------------------------------------
print("\n--- 2.4 Memoization Example - Factorial ---")

@lru_cache(maxsize=128)
def factorial_memoized(n: int) -> int:
    """Factorial with memoization."""
    if n < 2:
        return 1
    return n * factorial_memoized(n - 1)

# Example
result = factorial_memoized(5)
print(f"Factorial(5): {result}")


# ============================================================================
# 3. TABULATION (BOTTOM-UP)
# ============================================================================

print("\n" + "=" * 70)
print("3. TABULATION (BOTTOM-UP)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Fibonacci with Tabulation
# ----------------------------------------------------------------------------
print("\n--- 3.1 Fibonacci with Tabulation ---")

def fibonacci_tabulation(n: int) -> int:
    """
    Fibonacci using tabulation (bottom-up).
    Time: O(n), Space: O(n)
    """
    if n < 2:
        return n
    
    # Create table to store results
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    
    # Fill table bottom-up
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Example
result = fibonacci_tabulation(10)
print(f"Fibonacci(10) with tabulation: {result}")


def fibonacci_tabulation_optimized(n: int) -> int:
    """
    Fibonacci with space optimization.
    Only need last two values, not entire array.
    Time: O(n), Space: O(1)
    """
    if n < 2:
        return n
    
    prev2 = 0
    prev1 = 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1

# Example
result = fibonacci_tabulation_optimized(10)
print(f"\nFibonacci(10) optimized: {result}")


# ----------------------------------------------------------------------------
# 3.2 Tabulation vs Memoization
# ----------------------------------------------------------------------------
print("\n--- 3.2 Tabulation vs Memoization ---")
print("""
MEMOIZATION (Top-Down):
- Recursive approach
- Caches results as computed
- May not compute all subproblems
- More intuitive (natural recursion)
- Can have stack overflow for deep recursion

TABULATION (Bottom-Up):
- Iterative approach
- Fills table systematically
- Computes all subproblems
- More efficient (no recursion overhead)
- Better space optimization possible
""")


# ============================================================================
# 4. COMMON PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("4. COMMON PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Coin Change Problem
# ----------------------------------------------------------------------------
print("\n--- 4.1 Coin Change Problem ---")

def coin_change_memo(coins: List[int], amount: int, memo: Dict[int, int] = None) -> int:
    """
    Minimum coins needed using memoization.
    Time: O(amount * len(coins)), Space: O(amount)
    """
    if memo is None:
        memo = {}
    
    if amount in memo:
        return memo[amount]
    
    if amount == 0:
        return 0
    if amount < 0:
        return -1
    
    min_coins = float('inf')
    
    for coin in coins:
        result = coin_change_memo(coins, amount - coin, memo)
        if result != -1:
            min_coins = min(min_coins, result + 1)
    
    memo[amount] = min_coins if min_coins != float('inf') else -1
    return memo[amount]

# Example
coins = [1, 3, 4]
amount = 6
result = coin_change_memo(coins, amount)
print(f"Coins: {coins}, Amount: {amount}")
print(f"Minimum coins (memoization): {result}")


def coin_change_tabulation(coins: List[int], amount: int) -> int:
    """
    Minimum coins needed using tabulation.
    Time: O(amount * len(coins)), Space: O(amount)
    """
    # dp[i] = minimum coins needed for amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Example
coins = [1, 3, 4]
amount = 6
result = coin_change_tabulation(coins, amount)
print(f"\nCoins: {coins}, Amount: {amount}")
print(f"Minimum coins (tabulation): {result}")


# ----------------------------------------------------------------------------
# 4.2 Longest Common Subsequence (Basic)
# ----------------------------------------------------------------------------
print("\n--- 4.2 Longest Common Subsequence (Basic) ---")

def lcs_memo(s1: str, s2: str, i: int = None, j: int = None, memo: Dict = None) -> int:
    """
    Longest Common Subsequence using memoization.
    Time: O(m * n), Space: O(m * n)
    """
    if i is None:
        i = len(s1)
    if j is None:
        j = len(s2)
    if memo is None:
        memo = {}
    
    if (i, j) in memo:
        return memo[(i, j)]
    
    # Base case: empty strings
    if i == 0 or j == 0:
        result = 0
    # Characters match
    elif s1[i - 1] == s2[j - 1]:
        result = 1 + lcs_memo(s1, s2, i - 1, j - 1, memo)
    # Characters don't match
    else:
        result = max(
            lcs_memo(s1, s2, i - 1, j, memo),
            lcs_memo(s1, s2, i, j - 1, memo)
        )
    
    memo[(i, j)] = result
    return result

# Example
s1 = "ABCDGH"
s2 = "AEDFHR"
result = lcs_memo(s1, s2)
print(f"String1: '{s1}', String2: '{s2}'")
print(f"LCS length (memoization): {result}")


def lcs_tabulation(s1: str, s2: str) -> int:
    """
    Longest Common Subsequence using tabulation.
    Time: O(m * n), Space: O(m * n)
    """
    m, n = len(s1), len(s2)
    
    # dp[i][j] = LCS of s1[0:i] and s2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# Example
s1 = "ABCDGH"
s2 = "AEDFHR"
result = lcs_tabulation(s1, s2)
print(f"\nString1: '{s1}', String2: '{s2}'")
print(f"LCS length (tabulation): {result}")


# ----------------------------------------------------------------------------
# 4.3 Climbing Stairs
# ----------------------------------------------------------------------------
print("\n--- 4.3 Climbing Stairs ---")

def climb_stairs_memo(n: int, memo: Dict[int, int] = None) -> int:
    """
    Ways to climb n stairs (1 or 2 steps at a time) using memoization.
    Time: O(n), Space: O(n)
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 2:
        result = n
    else:
        result = climb_stairs_memo(n - 1, memo) + climb_stairs_memo(n - 2, memo)
    
    memo[n] = result
    return result

# Example
n = 5
result = climb_stairs_memo(n)
print(f"Ways to climb {n} stairs (memoization): {result}")


def climb_stairs_tabulation(n: int) -> int:
    """
    Ways to climb n stairs using tabulation.
    Time: O(n), Space: O(1) optimized
    """
    if n <= 2:
        return n
    
    prev2 = 1
    prev1 = 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1

# Example
n = 5
result = climb_stairs_tabulation(n)
print(f"Ways to climb {n} stairs (tabulation): {result}")


# ----------------------------------------------------------------------------
# 4.4 House Robber
# ----------------------------------------------------------------------------
print("\n--- 4.4 House Robber ---")

def house_robber_memo(nums: List[int], i: int = None, memo: Dict[int, int] = None) -> int:
    """
    Maximum money that can be robbed (can't rob adjacent houses).
    Time: O(n), Space: O(n)
    """
    if i is None:
        i = len(nums) - 1
    if memo is None:
        memo = {}
    
    if i in memo:
        return memo[i]
    
    if i < 0:
        result = 0
    else:
        # Rob current house or skip it
        result = max(
            house_robber_memo(nums, i - 2, memo) + nums[i],
            house_robber_memo(nums, i - 1, memo)
        )
    
    memo[i] = result
    return result

# Example
nums = [2, 7, 9, 3, 1]
result = house_robber_memo(nums)
print(f"Houses: {nums}")
print(f"Maximum money (memoization): {result}")


def house_robber_tabulation(nums: List[int]) -> int:
    """
    Maximum money using tabulation.
    Time: O(n), Space: O(1) optimized
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current
    
    return prev1

# Example
nums = [2, 7, 9, 3, 1]
result = house_robber_tabulation(nums)
print(f"\nHouses: {nums}")
print(f"Maximum money (tabulation): {result}")


# ============================================================================
# 5. MEMOIZATION VS TABULATION
# ============================================================================

print("\n" + "=" * 70)
print("5. MEMOIZATION VS TABULATION")
print("=" * 70)

print("""
MEMOIZATION (Top-Down):
✓ More intuitive (natural recursion)
✓ Only computes needed subproblems
✓ Easier to implement from recursive solution
✗ Recursion overhead
✗ Stack overflow risk for deep recursion
✗ May compute subproblems in any order

TABULATION (Bottom-Up):
✓ No recursion overhead
✓ Better space optimization possible
✓ Computes subproblems systematically
✓ No stack overflow risk
✗ Computes all subproblems (even unused ones)
✗ Less intuitive (need to figure out order)

WHEN TO USE EACH:
- Memoization: When recursion is natural, not all subproblems needed
- Tabulation: When need to optimize space, avoid recursion overhead
- Both work for most DP problems
- Choose based on problem and constraints
""")


# ============================================================================
# 6. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Fibonacci
print("\n--- Exercise 1: Fibonacci ---")
def fib_exercise(n: int) -> int:
    """Implement Fibonacci using tabulation."""
    if n < 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

result = fib_exercise(10)
print(f"Fibonacci(10): {result}")

# Exercise 2: Coin Change
print("\n--- Exercise 2: Coin Change ---")
def coin_change_exercise(coins: List[int], amount: int) -> int:
    """Implement coin change using tabulation."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

coins = [1, 2, 5]
amount = 11
result = coin_change_exercise(coins, amount)
print(f"Coins: {coins}, Amount: {amount}")
print(f"Minimum coins: {result}")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. DYNAMIC PROGRAMMING:
   - Solves overlapping subproblems
   - Stores results to avoid recomputation
   - Two approaches: Memoization and Tabulation

2. MEMOIZATION (Top-Down):
   - Recursive + caching
   - Use dictionary or lru_cache
   - More intuitive, natural recursion
   - Time: Usually O(n) or O(n²)
   - Space: O(n) for cache + recursion stack

3. TABULATION (Bottom-Up):
   - Iterative + table filling
   - Fill table systematically
   - Better space optimization
   - Time: Usually O(n) or O(n²)
   - Space: O(n) for table (can optimize to O(1))

4. COMMON PATTERNS:
   - Fibonacci: dp[i] = dp[i-1] + dp[i-2]
   - Coin Change: dp[i] = min(dp[i], dp[i-coin] + 1)
   - LCS: dp[i][j] = dp[i-1][j-1] + 1 if match, else max
   - Climbing Stairs: Similar to Fibonacci
   - House Robber: dp[i] = max(dp[i-1], dp[i-2] + nums[i])

5. IMPLEMENTATION STEPS:
   - Identify overlapping subproblems
   - Define state (what to store)
   - Write recurrence relation
   - Choose memoization or tabulation
   - Handle base cases
   - Optimize space if possible

6. OPTIMIZATION:
   - Space optimization: Only keep needed values
   - Example: Fibonacci only needs last 2 values
   - Convert 2D DP to 1D when possible
   - Use rolling arrays for space efficiency

7. WHEN TO USE DP:
   - Overlapping subproblems exist
   - Optimal substructure property
   - Recursive solution has repeated calculations
   - Need to optimize recursive solution

8. INTERVIEW TIPS:
   - Start with recursive solution
   - Identify repeated calculations
   - Add memoization or convert to tabulation
   - Explain time/space complexity
   - Discuss optimization opportunities
   - Show both approaches if time permits
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Dynamic Programming Guide Ready!")
    print("=" * 70)
