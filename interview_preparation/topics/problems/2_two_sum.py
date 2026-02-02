
"""_summary_
Given an array of integers nums and an integer target, return the indices i and j such that nums[i] + nums[j] == target and i != j.

You may assume that every input has exactly one pair of indices i and j that satisfy the condition.

Return the answer with the smaller index first.

Example 1:

Input: 
nums = [3,4,5,6], target = 7

Output: [0,1]
Explanation: nums[0] + nums[1] == 7, so we return [0, 1].

Example 2:

Input: nums = [4,5,6], target = 10

Output: [0,2]
Example 3:

Input: nums = [5,5], target = 10

Output: [0,1]
Constraints:

2 <= nums.length <= 1000
-10,000,000 <= nums[i] <= 10,000,000
-10,000,000 <= target <= 10,000,000
Only one valid answer exists.
"""

""" summary_
- iterate over the values and take their index
- for a value at a given indes, check the differnce from the target value
- get the difference from the target value and 
- check it the differece value exist in the 
- return index of value and diff value
    
"""

from typing import List
def twosum(nums, target) -> List[int]:
    
    index_val = {}
    for i, val in enumerate(nums):
        diff  = target-val
        if diff in index_val.keys():
            return [index_val[diff], i]
        
        index_val[val] = i        
    print(index_val)
nums = [3, 4, 5, 6]
target = 7

print(twosum(nums, target))

