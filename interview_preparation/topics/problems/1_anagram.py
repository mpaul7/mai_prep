"""
    Given two strings s and t, return true if the two strings are anagrams of each other, otherwise return false.

An anagram is a string that contains the exact same characters as another string, but the order of the characters can be different.

Example 1:

Input: s = "racecar", t = "carrace"

Output: true
Example 2:

Input: s = "jar", t = "jam"

Output: false
Constraints:

s and t consist of lowercase English letters.
"""

"""
Condistions

    If two strings are anagram, then
    they should have same number of characters
    shoudl have same frequency of characters
"""

from collections import defaultdict
def check_anagram(s, t) -> bool:
    if len(s) != len(t):
        return False
    
    frequency_s = defaultdict(int)
    frequency_t = defaultdict(int)
    
    for i in range(len(s)):

        frequency_s[s[i]] += 1
        frequency_t[t[i]] += 1

    return frequency_s == frequency_t
     
s = "racecar"
t = "carrace"

print(check_anagram(s, t))


