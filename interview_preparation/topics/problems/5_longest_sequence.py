"""ous element. The elements do not have to be consecutive in the original array.

You must write an algorithm that runs in O(n) time.

Example 1:

Input: nums = [2,20,4,10,3,4,5]

Output: 4
Explanation: The longest consecutive sequence is [2, 3, 4, 5].

Example 2:

Input: nums = [0,3,2,5,4,6,1,1]

Output: 7
Constraints:

0 <= nums.length <= 1000
-10^9 <= nums[i] <= 10^9
"""

def longest_sequence(nums):
    nums = set(nums)
    print(nums)
    longest = 0
    for n in nums:
        # print(n)
        if n - 1 not in nums:

            length = 0
            while n + length in nums:
                length += 1
            longest = max(longest, length)
    return longest

nums = [2,20,4,10,3,4,5]
print(longest_sequence(nums))