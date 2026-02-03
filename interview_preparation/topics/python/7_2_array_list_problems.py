"""
Python Array/List Problems - Interview Preparation
Topic 7.2: Array/List Problems

This module covers:
- Finding Elements: Maximum, minimum, duplicates
- Array Manipulation: Rotating, reversing, partitioning
- Subarray Problems: Maximum sum, contiguous subarrays
- Two Sum Variations: Finding pairs with given sum
"""

from collections import Counter, defaultdict
from typing import List, Tuple

# ============================================================================
# 1. FINDING ELEMENTS
# ============================================================================

print("=" * 70)
print("1. FINDING ELEMENTS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Finding Maximum and Minimum
# ----------------------------------------------------------------------------
print("\n--- 1.1 Finding Maximum and Minimum ---")

def find_maximum(numbers):
    """Find maximum value in list."""
    if not numbers:
        return None
    
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val

def find_minimum(numbers):
    """Find minimum value in list."""
    if not numbers:
        return None
    
    min_val = numbers[0]
    for num in numbers[1:]:
        if num < min_val:
            min_val = num
    return min_val

def find_max_min(numbers):
    """Find both maximum and minimum in single pass."""
    if not numbers:
        return None, None
    
    max_val = min_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
        if num < min_val:
            min_val = num
    return max_val, min_val

numbers = [3, 7, 2, 9, 1, 5]
print(f"Numbers: {numbers}")
print(f"Maximum: {find_maximum(numbers)}")
print(f"Minimum: {find_minimum(numbers)}")
max_val, min_val = find_max_min(numbers)
print(f"Max and Min: {max_val}, {min_val}")


# ----------------------------------------------------------------------------
# 1.2 Finding Maximum/Minimum with Index
# ----------------------------------------------------------------------------
print("\n--- 1.2 Finding Maximum/Minimum with Index ---")

def find_max_index(numbers):
    """Find index of maximum value."""
    if not numbers:
        return -1
    
    max_val = numbers[0]
    max_idx = 0
    
    for i in range(1, len(numbers)):
        if numbers[i] > max_val:
            max_val = numbers[i]
            max_idx = i
    
    return max_idx

def find_max_index_builtin(numbers):
    """Find index of maximum using built-in."""
    if not numbers:
        return -1
    return numbers.index(max(numbers))

numbers = [3, 7, 2, 9, 1, 5]
print(f"Index of maximum: {find_max_index(numbers)}")
print(f"Index of maximum (built-in): {find_max_index_builtin(numbers)}")


# ----------------------------------------------------------------------------
# 1.3 Finding Duplicates
# ----------------------------------------------------------------------------
print("\n--- 1.3 Finding Duplicates ---")

def find_duplicates_set(numbers):
    """Find duplicates using set - O(n) time."""
    seen = set()
    duplicates = []
    
    for num in numbers:
        if num in seen:
            if num not in duplicates:
                duplicates.append(num)
        else:
            seen.add(num)
    
    return duplicates

def find_duplicates_counter(numbers):
    """Find duplicates using Counter."""
    counts = Counter(numbers)
    return [num for num, count in counts.items() if count > 1]

def find_duplicates_sort(numbers):
    """Find duplicates using sorting - O(n log n) time."""
    if len(numbers) <= 1:
        return []
    
    sorted_nums = sorted(numbers)
    duplicates = []
    
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] == sorted_nums[i - 1]:
            if sorted_nums[i] not in duplicates:
                duplicates.append(sorted_nums[i])
    
    return duplicates

numbers = [1, 2, 3, 2, 4, 3, 5, 2]
print(f"Numbers: {numbers}")
print(f"Duplicates (set): {find_duplicates_set(numbers)}")
print(f"Duplicates (Counter): {find_duplicates_counter(numbers)}")
print(f"Duplicates (sort): {find_duplicates_sort(numbers)}")


# ----------------------------------------------------------------------------
# 1.4 Finding All Duplicates with Indices
# ----------------------------------------------------------------------------
print("\n--- 1.4 Finding All Duplicates with Indices ---")

def find_duplicate_indices(numbers):
    """Find all indices of duplicate values."""
    indices_map = defaultdict(list)
    
    for i, num in enumerate(numbers):
        indices_map[num].append(i)
    
    # Return only numbers with multiple occurrences
    return {num: indices for num, indices in indices_map.items() if len(indices) > 1}

numbers = [1, 2, 3, 2, 4, 3, 5, 2]
duplicate_indices = find_duplicate_indices(numbers)
print(f"Duplicate indices: {duplicate_indices}")


# ============================================================================
# 2. ARRAY MANIPULATION
# ============================================================================

print("\n" + "=" * 70)
print("2. ARRAY MANIPULATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Rotating Array
# ----------------------------------------------------------------------------
print("\n--- 2.1 Rotating Array ---")

def rotate_right(numbers, k):
    """
    Rotate array to the right by k positions.
    Time: O(n), Space: O(n)
    """
    if not numbers or k == 0:
        return numbers
    
    n = len(numbers)
    k = k % n  # Handle k > n
    
    return numbers[-k:] + numbers[:-k]

def rotate_left(numbers, k):
    """
    Rotate array to the left by k positions.
    Time: O(n), Space: O(n)
    """
    if not numbers or k == 0:
        return numbers
    
    n = len(numbers)
    k = k % n
    
    return numbers[k:] + numbers[:k]

def rotate_in_place(numbers, k):
    """
    Rotate array in place (modifies original).
    Time: O(n), Space: O(1)
    """
    if not numbers or k == 0:
        return
    
    n = len(numbers)
    k = k % n
    
    # Reverse entire array
    numbers.reverse()
    # Reverse first k elements
    numbers[:k] = reversed(numbers[:k])
    # Reverse remaining elements
    numbers[k:] = reversed(numbers[k:])

numbers = [1, 2, 3, 4, 5]
print(f"Original: {numbers}")
print(f"Rotate right by 2: {rotate_right(numbers.copy(), 2)}")
print(f"Rotate left by 2: {rotate_left(numbers.copy(), 2)}")

# In-place rotation
nums = numbers.copy()
rotate_in_place(nums, 2)
print(f"Rotate in place by 2: {nums}")


# ----------------------------------------------------------------------------
# 2.2 Reversing Array
# ----------------------------------------------------------------------------
print("\n--- 2.2 Reversing Array ---")

def reverse_array(numbers):
    """Reverse array - creates new array."""
    return numbers[::-1]

def reverse_array_in_place(numbers):
    """Reverse array in place."""
    left, right = 0, len(numbers) - 1
    while left < right:
        numbers[left], numbers[right] = numbers[right], numbers[left]
        left += 1
        right -= 1

numbers = [1, 2, 3, 4, 5]
print(f"Original: {numbers}")
print(f"Reversed (new): {reverse_array(numbers)}")

nums = numbers.copy()
reverse_array_in_place(nums)
print(f"Reversed (in place): {nums}")


# ----------------------------------------------------------------------------
# 2.3 Partitioning Array
# ----------------------------------------------------------------------------
print("\n--- 2.3 Partitioning Array ---")

def partition_by_value(numbers, pivot):
    """
    Partition array: elements <= pivot on left, > pivot on right.
    Returns new array.
    """
    left = [x for x in numbers if x <= pivot]
    right = [x for x in numbers if x > pivot]
    return left + right

def partition_in_place(numbers, pivot):
    """
    Partition array in place using two pointers.
    Modifies original array.
    """
    left = 0
    right = len(numbers) - 1
    
    while left < right:
        # Find element > pivot from left
        while left < len(numbers) and numbers[left] <= pivot:
            left += 1
        # Find element <= pivot from right
        while right >= 0 and numbers[right] > pivot:
            right -= 1
        
        if left < right:
            numbers[left], numbers[right] = numbers[right], numbers[left]
            left += 1
            right -= 1

numbers = [3, 1, 4, 2, 5, 1, 3]
pivot = 3
print(f"Original: {numbers}, Pivot: {pivot}")
print(f"Partitioned: {partition_by_value(numbers, pivot)}")

nums = numbers.copy()
partition_in_place(nums, pivot)
print(f"Partitioned (in place): {nums}")


# ----------------------------------------------------------------------------
# 2.4 Moving Zeros to End
# ----------------------------------------------------------------------------
print("\n--- 2.4 Moving Zeros to End ---")

def move_zeros_end(numbers):
    """Move all zeros to end, preserving order of non-zeros."""
    non_zeros = [x for x in numbers if x != 0]
    zeros = [0] * (len(numbers) - len(non_zeros))
    return non_zeros + zeros

def move_zeros_in_place(numbers):
    """Move zeros to end in place."""
    write_idx = 0
    
    # Write all non-zeros first
    for num in numbers:
        if num != 0:
            numbers[write_idx] = num
            write_idx += 1
    
    # Fill rest with zeros
    for i in range(write_idx, len(numbers)):
        numbers[i] = 0

numbers = [0, 1, 0, 3, 12]
print(f"Original: {numbers}")
print(f"Move zeros: {move_zeros_end(numbers)}")

nums = numbers.copy()
move_zeros_in_place(nums)
print(f"Move zeros (in place): {nums}")


# ============================================================================
# 3. SUBARRAY PROBLEMS
# ============================================================================

print("\n" + "=" * 70)
print("3. SUBARRAY PROBLEMS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Maximum Subarray Sum (Kadane's Algorithm)
# ----------------------------------------------------------------------------
print("\n--- 3.1 Maximum Subarray Sum (Kadane's Algorithm) ---")

def max_subarray_sum_kadane(numbers):
    """
    Find maximum sum of contiguous subarray using Kadane's algorithm.
    Time: O(n), Space: O(1)
    """
    if not numbers:
        return 0
    
    max_sum = current_sum = numbers[0]
    
    for num in numbers[1:]:
        # Either extend previous subarray or start new
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def max_subarray_sum_with_indices(numbers):
    """Find maximum subarray sum and return indices."""
    if not numbers:
        return 0, -1, -1
    
    max_sum = current_sum = numbers[0]
    start = end = 0
    temp_start = 0
    
    for i in range(1, len(numbers)):
        if current_sum < 0:
            current_sum = numbers[i]
            temp_start = i
        else:
            current_sum += numbers[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return max_sum, start, end

numbers = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(f"Numbers: {numbers}")
max_sum = max_subarray_sum_kadane(numbers)
print(f"Maximum subarray sum: {max_sum}")

max_sum, start, end = max_subarray_sum_with_indices(numbers)
print(f"Max sum: {max_sum}, Indices [{start}, {end}], Subarray: {numbers[start:end+1]}")


# ----------------------------------------------------------------------------
# 3.2 Maximum Subarray Sum (Brute Force)
# ----------------------------------------------------------------------------
print("\n--- 3.2 Maximum Subarray Sum (Brute Force) ---")

def max_subarray_sum_brute_force(numbers):
    """
    Brute force approach - check all subarrays.
    Time: O(n²), Space: O(1)
    """
    if not numbers:
        return 0
    
    max_sum = numbers[0]
    
    for i in range(len(numbers)):
        current_sum = 0
        for j in range(i, len(numbers)):
            current_sum += numbers[j]
            max_sum = max(max_sum, current_sum)
    
    return max_sum

numbers = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(f"Maximum subarray sum (brute force): {max_subarray_sum_brute_force(numbers)}")


# ----------------------------------------------------------------------------
# 3.3 Subarray with Given Sum
# ----------------------------------------------------------------------------
print("\n--- 3.3 Subarray with Given Sum ---")

def subarray_with_sum(numbers, target):
    """
    Find subarray with given sum using sliding window.
    Returns indices if found, None otherwise.
    Time: O(n), Space: O(1)
    """
    if not numbers:
        return None
    
    left = 0
    current_sum = 0
    
    for right in range(len(numbers)):
        current_sum += numbers[right]
        
        # Shrink window from left if sum exceeds target
        while current_sum > target and left <= right:
            current_sum -= numbers[left]
            left += 1
        
        if current_sum == target:
            return [left, right]
    
    return None

numbers = [1, 4, 20, 3, 10, 5]
target = 33
result = subarray_with_sum(numbers, target)
print(f"Subarray with sum {target}: {result}")
if result:
    print(f"Subarray: {numbers[result[0]:result[1]+1]}")


# ----------------------------------------------------------------------------
# 3.4 Count Subarrays with Given Sum
# ----------------------------------------------------------------------------
print("\n--- 3.4 Count Subarrays with Given Sum ---")

def count_subarrays_with_sum(numbers, target):
    """
    Count number of subarrays with given sum.
    Time: O(n), Space: O(n)
    """
    count = 0
    prefix_sum = 0
    sum_count = defaultdict(int)
    sum_count[0] = 1  # Empty subarray has sum 0
    
    for num in numbers:
        prefix_sum += num
        # Check if prefix_sum - target exists
        if prefix_sum - target in sum_count:
            count += sum_count[prefix_sum - target]
        sum_count[prefix_sum] += 1
    
    return count

numbers = [1, 1, 1]
target = 2
count = count_subarrays_with_sum(numbers, target)
print(f"Count of subarrays with sum {target}: {count}")


# ============================================================================
# 4. TWO SUM VARIATIONS
# ============================================================================

print("\n" + "=" * 70)
print("4. TWO SUM VARIATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Two Sum (Basic)
# ----------------------------------------------------------------------------
print("\n--- 4.1 Two Sum (Basic) ---")

def two_sum_basic(numbers, target):
    """
    Find two indices where values sum to target.
    Time: O(n), Space: O(n)
    """
    seen = {}
    
    for i, num in enumerate(numbers):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []

numbers = [2, 7, 11, 15]
target = 9
result = two_sum_basic(numbers, target)
print(f"Two sum: {result}")
if result:
    print(f"Values: {numbers[result[0]]}, {numbers[result[1]]}")


# ----------------------------------------------------------------------------
# 4.2 Two Sum (Sorted Array - Two Pointers)
# ----------------------------------------------------------------------------
print("\n--- 4.2 Two Sum (Sorted Array - Two Pointers) ---")

def two_sum_sorted(numbers, target):
    """
    Two sum for sorted array using two pointers.
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

numbers = [1, 2, 3, 4, 5, 6]
target = 7
result = two_sum_sorted(numbers, target)
print(f"Two sum (sorted): {result}")
if result:
    print(f"Values: {numbers[result[0]]}, {numbers[result[1]]}")


# ----------------------------------------------------------------------------
# 4.3 Two Sum - All Pairs
# ----------------------------------------------------------------------------
print("\n--- 4.3 Two Sum - All Pairs ---")

def two_sum_all_pairs(numbers, target):
    """
    Find all pairs of indices that sum to target.
    Time: O(n), Space: O(n)
    """
    pairs = []
    num_to_indices = defaultdict(list)
    
    for i, num in enumerate(numbers):
        complement = target - num
        if complement in num_to_indices:
            for j in num_to_indices[complement]:
                pairs.append([j, i])
        num_to_indices[num].append(i)
    
    return pairs

numbers = [1, 2, 3, 2, 4, 1]
target = 3
pairs = two_sum_all_pairs(numbers, target)
print(f"All pairs summing to {target}: {pairs}")


# ----------------------------------------------------------------------------
# 4.4 Three Sum
# ----------------------------------------------------------------------------
print("\n--- 4.4 Three Sum ---")

def three_sum(numbers, target):
    """
    Find three numbers that sum to target.
    Returns list of triplets (values, not indices).
    Time: O(n²), Space: O(1) excluding output
    """
    numbers.sort()
    triplets = []
    n = len(numbers)
    
    for i in range(n - 2):
        # Skip duplicates
        if i > 0 and numbers[i] == numbers[i - 1]:
            continue
        
        left, right = i + 1, n - 1
        
        while left < right:
            current_sum = numbers[i] + numbers[left] + numbers[right]
            
            if current_sum == target:
                triplets.append([numbers[i], numbers[left], numbers[right]])
                # Skip duplicates
                while left < right and numbers[left] == numbers[left + 1]:
                    left += 1
                while left < right and numbers[right] == numbers[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return triplets

numbers = [-1, 0, 1, 2, -1, -4]
target = 0
triplets = three_sum(numbers, target)
print(f"Three sum to {target}: {triplets}")


# ----------------------------------------------------------------------------
# 4.5 Four Sum
# ----------------------------------------------------------------------------
print("\n--- 4.5 Four Sum ---")

def four_sum(numbers, target):
    """
    Find four numbers that sum to target.
    Time: O(n³), Space: O(1) excluding output
    """
    numbers.sort()
    quadruplets = []
    n = len(numbers)
    
    for i in range(n - 3):
        if i > 0 and numbers[i] == numbers[i - 1]:
            continue
        
        for j in range(i + 1, n - 2):
            if j > i + 1 and numbers[j] == numbers[j - 1]:
                continue
            
            left, right = j + 1, n - 1
            
            while left < right:
                current_sum = numbers[i] + numbers[j] + numbers[left] + numbers[right]
                
                if current_sum == target:
                    quadruplets.append([numbers[i], numbers[j], numbers[left], numbers[right]])
                    while left < right and numbers[left] == numbers[left + 1]:
                        left += 1
                    while left < right and numbers[right] == numbers[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
    
    return quadruplets

numbers = [1, 0, -1, 0, -2, 2]
target = 0
quadruplets = four_sum(numbers, target)
print(f"Four sum to {target}: {quadruplets}")


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Find Second Largest
print("\n--- Exercise 1: Find Second Largest ---")
def find_second_largest(numbers):
    """Find second largest element."""
    if len(numbers) < 2:
        return None
    
    # Remove duplicates and sort
    unique = sorted(set(numbers), reverse=True)
    return unique[1] if len(unique) > 1 else None

numbers = [3, 7, 2, 9, 1, 5, 9]
print(f"Second largest: {find_second_largest(numbers)}")


# Exercise 2: Remove Duplicates Preserving Order
print("\n--- Exercise 2: Remove Duplicates Preserving Order ---")
def remove_duplicates_preserve_order(numbers):
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for num in numbers:
        if num not in seen:
            seen.add(num)
            result.append(num)
    return result

numbers = [1, 2, 3, 2, 4, 3, 5]
print(f"Remove duplicates: {remove_duplicates_preserve_order(numbers)}")


# Exercise 3: Product of Array Except Self
print("\n--- Exercise 3: Product of Array Except Self ---")
def product_except_self(numbers):
    """
    Return array where each element is product of all others.
    Time: O(n), Space: O(1) excluding output
    """
    n = len(numbers)
    result = [1] * n
    
    # Left products
    for i in range(1, n):
        result[i] = result[i - 1] * numbers[i - 1]
    
    # Right products
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= numbers[i]
    
    return result

numbers = [1, 2, 3, 4]
print(f"Product except self: {product_except_self(numbers)}")


# Exercise 4: Contains Duplicate
print("\n--- Exercise 4: Contains Duplicate ---")
def contains_duplicate(numbers):
    """Check if array contains any duplicate."""
    return len(numbers) != len(set(numbers))

numbers = [1, 2, 3, 1]
print(f"Contains duplicate: {contains_duplicate(numbers)}")


# Exercise 5: Best Time to Buy and Sell Stock
print("\n--- Exercise 5: Best Time to Buy and Sell Stock ---")
def max_profit(prices):
    """
    Find maximum profit from buying and selling stock.
    Can only buy once and sell once.
    """
    if not prices or len(prices) < 2:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)
    
    return max_profit

prices = [7, 1, 5, 3, 6, 4]
print(f"Max profit: {max_profit(prices)}")


# ============================================================================
# 6. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("6. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. FINDING ELEMENTS:
   - Maximum/Minimum: O(n) single pass
   - Duplicates: Use set (O(n)) or Counter
   - Consider edge cases: empty, single element

2. ARRAY MANIPULATION:
   - Rotating: Use slicing or reverse technique
   - Reversing: Slicing or two pointers
   - Partitioning: Two pointers for in-place
   - Consider in-place vs new array

3. SUBARRAY PROBLEMS:
   - Maximum sum: Kadane's algorithm O(n)
   - Sliding window for fixed size
   - Prefix sum for range queries
   - Hash map for count problems

4. TWO SUM VARIATIONS:
   - Unsorted: Hash map O(n)
   - Sorted: Two pointers O(n)
   - All pairs: Hash map with list of indices
   - Three/Four sum: Sort + two pointers

5. COMMON PATTERNS:
   - Two pointers: Sorted arrays
   - Sliding window: Subarray problems
   - Hash map: Fast lookups
   - Prefix sum: Range queries

6. BEST PRACTICES:
   - Handle empty arrays
   - Consider in-place vs new array
   - Optimize time/space complexity
   - Handle duplicates appropriately
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Array/List Problems Guide Ready!")
    print("=" * 70)
