"""
Greedy Algorithms - Interview Preparation
Topic 6.4: Greedy Algorithms

This module covers:
- Greedy Approach: Making locally optimal choices
- Common Patterns: Interval scheduling, coin change (simple)
"""

from typing import List, Tuple

# ============================================================================
# 1. UNDERSTANDING GREEDY ALGORITHMS
# ============================================================================

print("=" * 70)
print("1. UNDERSTANDING GREEDY ALGORITHMS")
print("=" * 70)

print("""
GREEDY ALGORITHM APPROACH:
- Makes locally optimal choice at each step
- Hopes these choices lead to globally optimal solution
- Doesn't reconsider previous choices
- Often simpler and faster than other approaches

KEY CHARACTERISTICS:
1. Greedy Choice Property: Local optimal choice leads to global optimum
2. Optimal Substructure: Problem can be broken into subproblems
3. No Backtracking: Once choice is made, it's final

WHEN GREEDY WORKS:
- Problem has greedy choice property
- Optimal substructure exists
- Can prove greedy choice is always optimal

WHEN GREEDY DOESN'T WORK:
- Need to consider future consequences
- Local optimal doesn't guarantee global optimal
- Need to backtrack or reconsider choices
""")


# ============================================================================
# 2. INTERVAL SCHEDULING
# ============================================================================

print("\n" + "=" * 70)
print("2. INTERVAL SCHEDULING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Activity Selection Problem
# ----------------------------------------------------------------------------
print("\n--- 2.1 Activity Selection Problem ---")

def activity_selection(activities: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Select maximum number of non-overlapping activities.
    Greedy: Always pick activity that ends earliest.
    Time: O(n log n) for sorting, O(n) for selection
    """
    # Sort by end time
    sorted_activities = sorted(activities, key=lambda x: x[1])
    
    selected = []
    last_end = -1
    
    for start, end in sorted_activities:
        # If activity doesn't overlap with last selected
        if start >= last_end:
            selected.append((start, end))
            last_end = end
    
    return selected

# Example
activities = [(1, 4), (3, 5), (0, 6), (5, 7), (8, 9), (5, 9)]
result = activity_selection(activities)
print(f"Activities: {activities}")
print(f"Selected activities: {result}")
print(f"Maximum activities: {len(result)}")


# ----------------------------------------------------------------------------
# 2.2 Meeting Rooms Problem
# ----------------------------------------------------------------------------
print("\n--- 2.2 Meeting Rooms Problem ---")

def min_meeting_rooms(intervals: List[Tuple[int, int]]) -> int:
    """
    Find minimum number of meeting rooms needed.
    Greedy: Track active meetings, need room when new meeting starts.
    Time: O(n log n), Space: O(n)
    """
    # Separate start and end times
    starts = sorted([interval[0] for interval in intervals])
    ends = sorted([interval[1] for interval in intervals])
    
    rooms = 0
    max_rooms = 0
    start_ptr = 0
    end_ptr = 0
    
    while start_ptr < len(starts):
        if starts[start_ptr] < ends[end_ptr]:
            # New meeting starts before one ends
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            start_ptr += 1
        else:
            # Meeting ends
            rooms -= 1
            end_ptr += 1
    
    return max_rooms

# Example
intervals = [(0, 30), (5, 10), (15, 20)]
result = min_meeting_rooms(intervals)
print(f"Intervals: {intervals}")
print(f"Minimum rooms needed: {result}")


# ----------------------------------------------------------------------------
# 2.3 Merge Intervals
# ----------------------------------------------------------------------------
print("\n--- 2.3 Merge Intervals ---")

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping intervals.
    Greedy: Sort by start, merge if overlaps with previous.
    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return []
    
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        
        # If current overlaps with last merged interval
        if current_start <= last_end:
            # Merge: update end time
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap, add new interval
            merged.append((current_start, current_end))
    
    return merged

# Example
intervals = [(1, 3), (2, 6), (8, 10), (15, 18)]
result = merge_intervals(intervals)
print(f"Intervals: {intervals}")
print(f"Merged: {result}")


# ============================================================================
# 3. COIN CHANGE (GREEDY)
# ============================================================================

print("\n" + "=" * 70)
print("3. COIN CHANGE (GREEDY)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Simple Coin Change (Greedy)
# ----------------------------------------------------------------------------
print("\n--- 3.1 Simple Coin Change (Greedy) ---")

def coin_change_greedy(coins: List[int], amount: int) -> int:
    """
    Find minimum coins needed using greedy approach.
    Greedy: Always use largest coin possible.
    NOTE: Only works for certain coin systems (e.g., US coins).
    Time: O(n), Space: O(1)
    """
    # Sort coins in descending order
    coins.sort(reverse=True)
    count = 0
    
    for coin in coins:
        if amount >= coin:
            num_coins = amount // coin
            count += num_coins
            amount -= num_coins * coin
        
        if amount == 0:
            break
    
    return count if amount == 0 else -1

# Example - Works for US coin system
coins = [1, 5, 10, 25]
amount = 67
result = coin_change_greedy(coins, amount)
print(f"Coins: {coins}, Amount: {amount}")
print(f"Minimum coins (greedy): {result}")

# Example - Greedy doesn't work optimally
coins = [1, 3, 4]
amount = 6
result = coin_change_greedy(coins, amount)
print(f"\nCoins: {coins}, Amount: {amount}")
print(f"Greedy result: {result} (uses 4+1+1 = 3 coins)")
print(f"Optimal: 3+3 = 2 coins (greedy fails!)")


# ----------------------------------------------------------------------------
# 3.2 Coin Change - Count Ways (Greedy doesn't work)
# ----------------------------------------------------------------------------
print("\n--- 3.2 Coin Change - When Greedy Fails ---")
print("""
IMPORTANT: Greedy coin change only works for certain coin systems.
For general coin change problems, need dynamic programming.

Greedy works for: US coins (1, 5, 10, 25) - canonical coin system
Greedy fails for: Non-canonical systems like [1, 3, 4]
""")


# ============================================================================
# 4. OTHER COMMON GREEDY PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("4. OTHER COMMON GREEDY PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Maximum Profit (Stock Trading)
# ----------------------------------------------------------------------------
print("\n--- 4.1 Maximum Profit (Stock Trading) ---")

def max_profit(prices: List[int]) -> int:
    """
    Buy and sell stock once for maximum profit.
    Greedy: Track minimum price seen, calculate profit at each day.
    Time: O(n), Space: O(1)
    """
    if not prices:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices[1:]:
        # Update minimum price seen so far
        min_price = min(min_price, price)
        # Calculate profit if selling today
        profit = price - min_price
        max_profit = max(max_profit, profit)
    
    return max_profit

# Example
prices = [7, 1, 5, 3, 6, 4]
result = max_profit(prices)
print(f"Prices: {prices}")
print(f"Maximum profit: {result}")


def max_profit_multiple_transactions(prices: List[int]) -> int:
    """
    Buy and sell stock multiple times for maximum profit.
    Greedy: Buy before every price increase, sell before every decrease.
    Time: O(n), Space: O(1)
    """
    if not prices:
        return 0
    
    profit = 0
    
    for i in range(1, len(prices)):
        # If price increases, add to profit
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    
    return profit

# Example
prices = [7, 1, 5, 3, 6, 4]
result = max_profit_multiple_transactions(prices)
print(f"\nPrices: {prices}")
print(f"Maximum profit (multiple transactions): {result}")


# ----------------------------------------------------------------------------
# 4.2 Jump Game
# ----------------------------------------------------------------------------
print("\n--- 4.2 Jump Game ---")

def can_jump(nums: List[int]) -> bool:
    """
    Determine if can reach last index.
    Greedy: Track farthest reachable position.
    Time: O(n), Space: O(1)
    """
    farthest = 0
    
    for i in range(len(nums)):
        # If current position is beyond farthest reachable
        if i > farthest:
            return False
        
        # Update farthest reachable position
        farthest = max(farthest, i + nums[i])
        
        # Early exit if reached end
        if farthest >= len(nums) - 1:
            return True
    
    return True

# Example
nums = [2, 3, 1, 1, 4]
result = can_jump(nums)
print(f"Array: {nums}")
print(f"Can jump to end: {result}")


def min_jumps(nums: List[int]) -> int:
    """
    Find minimum jumps to reach last index.
    Greedy: Track current reach and next reach.
    Time: O(n), Space: O(1)
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_reach = 0
    next_reach = 0
    
    for i in range(len(nums) - 1):
        # Update farthest we can reach from current position
        next_reach = max(next_reach, i + nums[i])
        
        # If reached current reach limit, need to jump
        if i == current_reach:
            jumps += 1
            current_reach = next_reach
            
            # If can reach end, return
            if current_reach >= len(nums) - 1:
                return jumps
    
    return jumps

# Example
nums = [2, 3, 1, 1, 4]
result = min_jumps(nums)
print(f"\nArray: {nums}")
print(f"Minimum jumps: {result}")


# ----------------------------------------------------------------------------
# 4.3 Assign Cookies
# ----------------------------------------------------------------------------
print("\n--- 4.3 Assign Cookies ---")

def assign_cookies(greed: List[int], cookies: List[int]) -> int:
    """
    Assign cookies to children to maximize satisfied children.
    Greedy: Assign smallest cookie to smallest greed.
    Time: O(n log n) for sorting, O(n) for assignment
    """
    greed.sort()
    cookies.sort()
    
    child_idx = 0
    cookie_idx = 0
    satisfied = 0
    
    while child_idx < len(greed) and cookie_idx < len(cookies):
        # If cookie satisfies child's greed
        if cookies[cookie_idx] >= greed[child_idx]:
            satisfied += 1
            child_idx += 1
        
        cookie_idx += 1
    
    return satisfied

# Example
greed = [1, 2, 3]
cookies = [1, 1]
result = assign_cookies(greed, cookies)
print(f"Greed: {greed}, Cookies: {cookies}")
print(f"Satisfied children: {result}")


# ----------------------------------------------------------------------------
# 4.4 Non-overlapping Intervals
# ----------------------------------------------------------------------------
print("\n--- 4.4 Non-overlapping Intervals ---")

def erase_overlap_intervals(intervals: List[Tuple[int, int]]) -> int:
    """
    Find minimum intervals to remove to make non-overlapping.
    Greedy: Keep intervals that end earliest.
    Time: O(n log n), Space: O(1)
    """
    if not intervals:
        return 0
    
    # Sort by end time
    sorted_intervals = sorted(intervals, key=lambda x: x[1])
    
    count = 0
    last_end = sorted_intervals[0][1]
    
    for start, end in sorted_intervals[1:]:
        # If overlaps with last kept interval
        if start < last_end:
            count += 1  # Remove this interval
        else:
            last_end = end  # Keep this interval
    
    return count

# Example
intervals = [(1, 2), (2, 3), (3, 4), (1, 3)]
result = erase_overlap_intervals(intervals)
print(f"Intervals: {intervals}")
print(f"Minimum to remove: {result}")


# ----------------------------------------------------------------------------
# 4.5 Partition Labels
# ----------------------------------------------------------------------------
print("\n--- 4.5 Partition Labels ---")

def partition_labels(s: str) -> List[int]:
    """
    Partition string into as many parts as possible.
    Greedy: Track last occurrence of each character.
    Time: O(n), Space: O(1) - fixed alphabet
    """
    # Find last occurrence of each character
    last_occurrence = {}
    for i, char in enumerate(s):
        last_occurrence[char] = i
    
    result = []
    start = 0
    end = 0
    
    for i, char in enumerate(s):
        # Extend partition to include last occurrence of current char
        end = max(end, last_occurrence[char])
        
        # If reached end of partition
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    
    return result

# Example
s = "ababcbacadefegdehijhklij"
result = partition_labels(s)
print(f"String: '{s}'")
print(f"Partition sizes: {result}")


# ============================================================================
# 5. GREEDY VS OTHER APPROACHES
# ============================================================================

print("\n" + "=" * 70)
print("5. GREEDY VS OTHER APPROACHES")
print("=" * 70)

print("""
WHEN TO USE GREEDY:
✓ Problem has optimal substructure
✓ Greedy choice property holds
✓ Can prove correctness
✓ Need efficient solution
✓ Examples: Activity selection, interval scheduling, some coin systems

WHEN NOT TO USE GREEDY:
✗ Need to consider all possibilities
✗ Local optimal ≠ global optimal
✗ Need to backtrack
✗ Examples: General coin change, longest path, some optimization problems

GREEDY VS DYNAMIC PROGRAMMING:
- Greedy: Make choice and never reconsider
- DP: Consider all possibilities, may reconsider
- Greedy: Usually faster, simpler
- DP: More general, handles more problems

GREEDY VS BRUTE FORCE:
- Greedy: O(n log n) or O(n) typically
- Brute Force: Exponential time
- Greedy: Single pass or sorted + pass
- Brute Force: Try all combinations
""")


# ============================================================================
# 6. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Activity Selection
print("\n--- Exercise 1: Activity Selection ---")
activities = [(1, 2), (3, 4), (0, 6), (5, 7), (8, 9), (5, 9)]
result = activity_selection(activities)
print(f"Activities: {activities}")
print(f"Selected: {result}")

# Exercise 2: Maximum Profit
print("\n--- Exercise 2: Maximum Profit ---")
prices = [7, 6, 4, 3, 1]
result = max_profit(prices)
print(f"Prices: {prices}")
print(f"Maximum profit: {result}")

# Exercise 3: Can Jump
print("\n--- Exercise 3: Can Jump ---")
nums = [3, 2, 1, 0, 4]
result = can_jump(nums)
print(f"Array: {nums}")
print(f"Can jump: {result}")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. GREEDY APPROACH:
   - Make locally optimal choice at each step
   - Hope it leads to globally optimal solution
   - Don't reconsider previous choices
   - Often simpler and faster

2. KEY PROPERTIES:
   - Greedy Choice Property: Local optimal → global optimal
   - Optimal Substructure: Problem breaks into subproblems
   - No Backtracking: Choices are final

3. COMMON PATTERNS:
   - Interval Scheduling: Sort by end time, pick earliest ending
   - Coin Change: Use largest coin first (works for canonical systems)
   - Stock Trading: Track min price, calculate profit
   - Jump Game: Track farthest reachable position

4. WHEN GREEDY WORKS:
   - Problem has greedy choice property
   - Optimal substructure exists
   - Can prove correctness
   - Examples: Activity selection, interval problems

5. WHEN GREEDY FAILS:
   - Need to consider future consequences
   - Local optimal ≠ global optimal
   - Need backtracking
   - Examples: General coin change, longest path

6. IMPLEMENTATION STEPS:
   - Sort if needed (often by end time or value)
   - Make greedy choice at each step
   - Update state based on choice
   - Continue until done

7. TIME COMPLEXITY:
   - Often O(n log n) due to sorting
   - Sometimes O(n) if no sorting needed
   - Usually better than DP or brute force

8. INTERVIEW TIPS:
   - Explain why greedy works (prove correctness)
   - Consider edge cases (empty, single element)
   - Discuss when greedy might not work
   - Compare with other approaches
   - Show understanding of trade-offs
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Greedy Algorithms Guide Ready!")
    print("=" * 70)
