"""_summary_
Given an array of strings strs, group all anagrams together into sublists. You may return the output in any order.

An anagram is a string that contains the exact same characters as another string, but the order of the characters can be different.

Example 1:

Input: strs = ["act","pots","tops","cat","stop","hat"]

Output: [["hat"],["act", "cat"],["stop", "pots", "tops"]]
Example 2:

Input: strs = ["x"]

Output: [["x"]]
Example 3:

Input: strs = [""]

Output: [[""]]
Constraints:

1 <= strs.length <= 1000.
0 <= strs[i].length <= 100
strs[i] is made up of lowercase English letters.
"""
"""
        to check for anagram
        - anagram the nubmer of characters are same
        - frequecny of characters is also same

        if anagram: 

            save the current word and its anagram in a hashmap, 
            - print the values of all the keys
"""

from collections import defaultdict
from typing import List
res = defaultdict(list)
def group_anagrams(strs: List[str]) -> List[List[str]]:
        
        for s in strs:
            count = [0] * 26
            # print(count)
            for c in s:
                count[ord(c) - ord('a')] += 1
            print(tuple(count))
            res[tuple(count)].append(s)
        return list(res.values())
        
    
strs = ["act","pots","tops","cat","stop","hat"]

print(group_anagrams(strs))
print(res)