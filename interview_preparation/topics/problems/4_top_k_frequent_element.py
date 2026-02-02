"""_summary_
    Given an integer array nums and an integer k, return the k most frequent elements within the array.

The test cases are generated such that the answer is always unique.

You may return the output in any order.

Example 1:

Input: nums = [1,2,2,3,3,3, ], k = 2

Output: [2,3]
Example 2:

Input: nums = [7,7], k = 1

Output: [7]
Constraints:

1 <= nums.length <= 10^4.
-1000 <= nums[i] <= 1000
1 <= k <= number of distinct elements in nums.
"""
    
    
"""
    - count the frequency of each element in the array
    
"""
    
    
def top_k_frequent(nums, k):
    count = {}
    freq = [[] for i in range(len(nums) + 1)]
    for n in nums:
        count[n] = 1 + count.get(n, 0)
    print(count)
    for n, c in count.items():
        freq[c].append(n)
    print(freq)
    res = []
    for i in range(len(freq) - 1, 0, -1):
        # print(i)
        for n in freq[i]:
            print(n)
            res.append(n)
            # if len(res) == k: return res
    return res

nums = [1,2,2,3,3,3]
k = 2
print(top_k_frequent(nums, k))