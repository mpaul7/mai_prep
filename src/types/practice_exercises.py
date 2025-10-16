"""
Practice Exercises for Python Types - Interview Preparation

This module contains additional practice exercises focusing on Python types,
collections, and iteration patterns commonly tested in data science interviews.
"""

from typing import List, Dict, Set, Tuple, Union, Any, Optional
from collections import Counter, defaultdict, deque
import re


class PythonTypesExercises:
    """
    Additional practice exercises for Python types and collections.
    """
    
    def __init__(self):
        """Initialize the exercises class."""
        pass
    
    # ========== STRING EXERCISES ==========
    
    def reverse_words_in_string(self, s: str) -> str:
        """
        Reverse the order of words in a string while preserving spaces.
        
        Example: "  hello   world  " -> "  world   hello  "
        """
        # Split by spaces but preserve the space pattern
        words = s.split(' ')
        # Filter out empty strings (which represent spaces)
        actual_words = [word for word in words if word]
        # Reverse the actual words
        actual_words.reverse()
        
        # Reconstruct with original spacing
        result = []
        word_index = 0
        
        for part in words:
            if part:  # If it's a word
                if word_index < len(actual_words):
                    result.append(actual_words[word_index])
                    word_index += 1
            else:  # If it's a space
                result.append('')
        
        return ' '.join(result)
    
    def is_anagram(self, s1: str, s2: str) -> bool:
        """
        Check if two strings are anagrams (ignoring case and spaces).
        
        Example: "listen", "silent" -> True
        """
        # Remove spaces and convert to lowercase
        clean_s1 = ''.join(s1.lower().split())
        clean_s2 = ''.join(s2.lower().split())
        
        # Check if sorted characters are equal
        return sorted(clean_s1) == sorted(clean_s2)
    
    def longest_common_substring(self, s1: str, s2: str) -> str:
        """
        Find the longest common substring between two strings.
        
        Example: "abcdxyz", "xyzabcd" -> "abcd" or "xyz"
        """
        longest = ""
        
        for i in range(len(s1)):
            for j in range(i + 1, len(s1) + 1):
                substring = s1[i:j]
                if substring in s2 and len(substring) > len(longest):
                    longest = substring
        
        return longest
    
    def compress_string(self, s: str) -> str:
        """
        Compress a string using run-length encoding.
        
        Example: "aabcccccaaa" -> "a2b1c5a3"
        """
        if not s:
            return ""
        
        compressed = []
        current_char = s[0]
        count = 1
        
        for i in range(1, len(s)):
            if s[i] == current_char:
                count += 1
            else:
                compressed.append(f"{current_char}{count}")
                current_char = s[i]
                count = 1
        
        # Add the last group
        compressed.append(f"{current_char}{count}")
        
        result = ''.join(compressed)
        return result if len(result) < len(s) else s
    
    def validate_parentheses(self, s: str) -> bool:
        """
        Check if parentheses, brackets, and braces are balanced.
        
        Example: "({[]})" -> True, "({[}])" -> False
        """
        stack = []
        mapping = {')': '(', ']': '[', '}': '{'}
        
        for char in s:
            if char in mapping:  # Closing bracket
                if not stack or stack.pop() != mapping[char]:
                    return False
            elif char in '([{':  # Opening bracket
                stack.append(char)
        
        return len(stack) == 0
    
    # ========== LIST EXERCISES ==========
    
    def find_missing_number(self, nums: List[int]) -> int:
        """
        Find the missing number in a sequence from 0 to n.
        
        Example: [3, 0, 1] -> 2
        """
        n = len(nums)
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(nums)
        return expected_sum - actual_sum
    
    def merge_sorted_lists(self, list1: List[int], list2: List[int]) -> List[int]:
        """
        Merge two sorted lists into one sorted list.
        
        Example: [1, 3, 5], [2, 4, 6] -> [1, 2, 3, 4, 5, 6]
        """
        merged = []
        i = j = 0
        
        while i < len(list1) and j < len(list2):
            if list1[i] <= list2[j]:
                merged.append(list1[i])
                i += 1
            else:
                merged.append(list2[j])
                j += 1
        
        # Add remaining elements
        merged.extend(list1[i:])
        merged.extend(list2[j:])
        
        return merged
    
    def find_duplicates(self, nums: List[int]) -> List[int]:
        """
        Find all duplicates in a list.
        
        Example: [4, 3, 2, 7, 8, 2, 3, 1] -> [2, 3]
        """
        seen = set()
        duplicates = set()
        
        for num in nums:
            if num in seen:
                duplicates.add(num)
            else:
                seen.add(num)
        
        return list(duplicates)
    
    def rotate_list(self, nums: List[int], k: int) -> List[int]:
        """
        Rotate list to the right by k steps.
        
        Example: [1, 2, 3, 4, 5], k=2 -> [4, 5, 1, 2, 3]
        """
        if not nums or k == 0:
            return nums
        
        k = k % len(nums)  # Handle k > len(nums)
        return nums[-k:] + nums[:-k]
    
    def two_sum(self, nums: List[int], target: int) -> List[Tuple[int, int]]:
        """
        Find all pairs of indices where nums[i] + nums[j] = target.
        
        Example: [2, 7, 11, 15], target=9 -> [(0, 1)]
        """
        num_to_indices = {}
        pairs = []
        
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_to_indices:
                for j in num_to_indices[complement]:
                    pairs.append((j, i))
            
            if num not in num_to_indices:
                num_to_indices[num] = []
            num_to_indices[num].append(i)
        
        return pairs
    
    # ========== DICTIONARY EXERCISES ==========
    
    def group_anagrams(self, words: List[str]) -> Dict[str, List[str]]:
        """
        Group words that are anagrams of each other.
        
        Example: ["eat", "tea", "tan", "ate", "nat", "bat"]
        -> {"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"], "abt": ["bat"]}
        """
        anagram_groups = defaultdict(list)
        
        for word in words:
            # Sort characters to create a key
            sorted_word = ''.join(sorted(word.lower()))
            anagram_groups[sorted_word].append(word)
        
        return dict(anagram_groups)
    
    def word_frequency(self, text: str) -> Dict[str, int]:
        """
        Count the frequency of each word in a text (case-insensitive).
        
        Example: "The quick brown fox jumps over the lazy dog"
        -> {"the": 2, "quick": 1, "brown": 1, ...}
        """
        # Remove punctuation and convert to lowercase
        words = re.findall(r'\b\w+\b', text.lower())
        return dict(Counter(words))
    
    def merge_dictionaries(self, dict1: Dict[str, int], dict2: Dict[str, int]) -> Dict[str, int]:
        """
        Merge two dictionaries, summing values for common keys.
        
        Example: {"a": 1, "b": 2}, {"b": 3, "c": 4} -> {"a": 1, "b": 5, "c": 4}
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            result[key] = result.get(key, 0) + value
        
        return result
    
    def invert_dictionary(self, d: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Invert a dictionary, making values keys and keys values.
        Handle multiple keys with the same value.
        
        Example: {"a": "1", "b": "2", "c": "1"} -> {"1": ["a", "c"], "2": ["b"]}
        """
        inverted = defaultdict(list)
        
        for key, value in d.items():
            inverted[value].append(key)
        
        return dict(inverted)
    
    def nested_dict_access(self, data: Dict, path: str, default=None) -> Any:
        """
        Access nested dictionary values using dot notation.
        
        Example: {"user": {"profile": {"name": "Alice"}}}, "user.profile.name" -> "Alice"
        """
        keys = path.split('.')
        current = data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    # ========== SET EXERCISES ==========
    
    def find_intersection(self, lists: List[List[int]]) -> Set[int]:
        """
        Find elements that appear in all lists.
        
        Example: [[1, 2, 3], [2, 3, 4], [3, 4, 5]] -> {3}
        """
        if not lists:
            return set()
        
        result = set(lists[0])
        for lst in lists[1:]:
            result &= set(lst)
        
        return result
    
    def find_unique_elements(self, lists: List[List[int]]) -> Set[int]:
        """
        Find elements that appear in exactly one list.
        
        Example: [[1, 2, 3], [2, 3, 4], [3, 4, 5]] -> {1, 5}
        """
        element_count = Counter()
        
        for lst in lists:
            for element in set(lst):  # Use set to avoid counting duplicates within same list
                element_count[element] += 1
        
        return {element for element, count in element_count.items() if count == 1}
    
    def symmetric_difference_multiple(self, sets: List[Set[int]]) -> Set[int]:
        """
        Find symmetric difference across multiple sets.
        
        Example: [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}] -> {1, 5}
        """
        if not sets:
            return set()
        
        result = sets[0]
        for s in sets[1:]:
            result ^= s
        
        return result
    
    def power_set(self, s: Set[int]) -> List[Set[int]]:
        """
        Generate all possible subsets (power set) of a set.
        
        Example: {1, 2} -> [set(), {1}, {2}, {1, 2}]
        """
        s_list = list(s)
        power_set_list = []
        
        for i in range(2**len(s_list)):
            subset = set()
            for j in range(len(s_list)):
                if i & (1 << j):
                    subset.add(s_list[j])
            power_set_list.append(subset)
        
        return power_set_list
    
    # ========== TUPLE EXERCISES ==========
    
    def sort_tuples_by_element(self, tuples: List[Tuple[str, int]], index: int) -> List[Tuple[str, int]]:
        """
        Sort a list of tuples by a specific element index.
        
        Example: [("Alice", 25), ("Bob", 20), ("Charlie", 30)], index=1
        -> [("Bob", 20), ("Alice", 25), ("Charlie", 30)]
        """
        return sorted(tuples, key=lambda x: x[index])
    
    def tuple_to_dict(self, tuples: List[Tuple[str, int]]) -> Dict[str, int]:
        """
        Convert a list of tuples to a dictionary.
        
        Example: [("a", 1), ("b", 2), ("c", 3)] -> {"a": 1, "b": 2, "c": 3}
        """
        return dict(tuples)
    
    def find_closest_points(self, points: List[Tuple[int, int]], 
                          reference: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find the k closest points to a reference point.
        
        Example: [(1, 1), (2, 2), (3, 3)], reference=(0, 0), k=2
        -> [(1, 1), (2, 2)]
        """
        def distance_squared(p1, p2):
            return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        
        return sorted(points, key=lambda p: distance_squared(p, reference))
    
    def group_tuples_by_first(self, tuples: List[Tuple[str, int]]) -> Dict[str, List[int]]:
        """
        Group tuples by their first element.
        
        Example: [("a", 1), ("b", 2), ("a", 3)] -> {"a": [1, 3], "b": [2]}
        """
        groups = defaultdict(list)
        
        for first, second in tuples:
            groups[first].append(second)
        
        return dict(groups)
    
    # ========== ITERATION EXERCISES ==========
    
    def flatten_nested_list(self, nested_list: List[Union[int, List]]) -> List[int]:
        """
        Flatten a nested list of arbitrary depth.
        
        Example: [1, [2, 3], [4, [5, 6]]] -> [1, 2, 3, 4, 5, 6]
        """
        result = []
        
        for item in nested_list:
            if isinstance(item, list):
                result.extend(self.flatten_nested_list(item))
            else:
                result.append(item)
        
        return result
    
    def chunk_list(self, lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """
        Split a list into chunks of specified size.
        
        Example: [1, 2, 3, 4, 5, 6, 7], chunk_size=3 -> [[1, 2, 3], [4, 5, 6], [7]]
        """
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
    def pairwise_iteration(self, lst: List[Any]) -> List[Tuple[Any, Any]]:
        """
        Create pairs of consecutive elements.
        
        Example: [1, 2, 3, 4] -> [(1, 2), (2, 3), (3, 4)]
        """
        return [(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]
    
    def sliding_window(self, lst: List[Any], window_size: int) -> List[List[Any]]:
        """
        Create a sliding window view of the list.
        
        Example: [1, 2, 3, 4, 5], window_size=3 -> [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        """
        return [lst[i:i + window_size] for i in range(len(lst) - window_size + 1)]
    
    def transpose_matrix(self, matrix: List[List[Any]]) -> List[List[Any]]:
        """
        Transpose a matrix (swap rows and columns).
        
        Example: [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        """
        if not matrix or not matrix[0]:
            return []
        
        return list(map(list, zip(*matrix)))
    
    # ========== ADVANCED EXERCISES ==========
    
    def lru_cache_simulation(self, capacity: int, operations: List[Tuple[str, int, Optional[int]]]) -> List[Optional[int]]:
        """
        Simulate an LRU (Least Recently Used) cache.
        
        Operations: [("get", key), ("put", key, value)]
        Returns: List of results for get operations (None for put operations)
        """
        cache = {}
        order = deque()
        results = []
        
        for operation in operations:
            if operation[0] == "get":
                key = operation[1]
                if key in cache:
                    # Move to end (most recently used)
                    order.remove(key)
                    order.append(key)
                    results.append(cache[key])
                else:
                    results.append(None)
            
            elif operation[0] == "put":
                key, value = operation[1], operation[2]
                
                if key in cache:
                    # Update existing key
                    cache[key] = value
                    order.remove(key)
                    order.append(key)
                else:
                    # Add new key
                    if len(cache) >= capacity:
                        # Remove least recently used
                        lru_key = order.popleft()
                        del cache[lru_key]
                    
                    cache[key] = value
                    order.append(key)
                
                results.append(None)
        
        return results
    
    def json_path_finder(self, data: Dict, target_value: Any) -> List[str]:
        """
        Find all JSON paths that lead to a target value.
        
        Example: {"a": {"b": 1, "c": {"d": 1}}, "e": 1}, target=1
        -> ["a.b", "a.c.d", "e"]
        """
        paths = []
        
        def dfs(obj, current_path):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    if value == target_value:
                        paths.append(new_path)
                    else:
                        dfs(value, new_path)
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                    if value == target_value:
                        paths.append(new_path)
                    else:
                        dfs(value, new_path)
        
        dfs(data, "")
        return paths
    
    def memory_efficient_counter(self, iterable) -> Dict[Any, int]:
        """
        Count elements in an iterable without loading everything into memory.
        Uses a generator approach for large datasets.
        """
        counter = defaultdict(int)
        
        for item in iterable:
            counter[item] += 1
        
        return dict(counter)


def run_practice_exercises():
    """Run all practice exercises with test cases."""
    exercises = PythonTypesExercises()
    
    print("=== Python Types Practice Exercises ===\n")
    
    # String exercises
    print("String Exercises:")
    print(f"1. Reverse words: '{exercises.reverse_words_in_string('  hello   world  ')}'")
    print(f"2. Is anagram: {exercises.is_anagram('listen', 'silent')}")
    print(f"3. Longest common substring: '{exercises.longest_common_substring('abcdxyz', 'xyzabcd')}'")
    print(f"4. Compress string: '{exercises.compress_string('aabcccccaaa')}'")
    print(f"5. Valid parentheses: {exercises.validate_parentheses('({[]})')}")
    
    # List exercises
    print(f"\nList Exercises:")
    print(f"1. Missing number: {exercises.find_missing_number([3, 0, 1])}")
    print(f"2. Merge sorted: {exercises.merge_sorted_lists([1, 3, 5], [2, 4, 6])}")
    print(f"3. Find duplicates: {exercises.find_duplicates([4, 3, 2, 7, 8, 2, 3, 1])}")
    print(f"4. Rotate list: {exercises.rotate_list([1, 2, 3, 4, 5], 2)}")
    print(f"5. Two sum: {exercises.two_sum([2, 7, 11, 15], 9)}")
    
    # Dictionary exercises
    print(f"\nDictionary Exercises:")
    print(f"1. Group anagrams: {exercises.group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat'])}")
    print(f"2. Word frequency: {exercises.word_frequency('The quick brown fox jumps over the lazy dog')}")
    print(f"3. Merge dicts: {exercises.merge_dictionaries({'a': 1, 'b': 2}, {'b': 3, 'c': 4})}")
    print(f"4. Invert dict: {exercises.invert_dictionary({'a': '1', 'b': '2', 'c': '1'})}")
    
    # Set exercises
    print(f"\nSet Exercises:")
    print(f"1. Find intersection: {exercises.find_intersection([[1, 2, 3], [2, 3, 4], [3, 4, 5]])}")
    print(f"2. Unique elements: {exercises.find_unique_elements([[1, 2, 3], [2, 3, 4], [3, 4, 5]])}")
    print(f"3. Power set: {exercises.power_set({1, 2})}")
    
    # Tuple exercises
    print(f"\nTuple Exercises:")
    tuples_data = [("Alice", 25), ("Bob", 20), ("Charlie", 30)]
    print(f"1. Sort by age: {exercises.sort_tuples_by_element(tuples_data, 1)}")
    print(f"2. Tuple to dict: {exercises.tuple_to_dict([('a', 1), ('b', 2), ('c', 3)])}")
    
    # Iteration exercises
    print(f"\nIteration Exercises:")
    print(f"1. Flatten nested: {exercises.flatten_nested_list([1, [2, 3], [4, [5, 6]]])}")
    print(f"2. Chunk list: {exercises.chunk_list([1, 2, 3, 4, 5, 6, 7], 3)}")
    print(f"3. Pairwise: {exercises.pairwise_iteration([1, 2, 3, 4])}")
    print(f"4. Sliding window: {exercises.sliding_window([1, 2, 3, 4, 5], 3)}")
    print(f"5. Transpose matrix: {exercises.transpose_matrix([[1, 2, 3], [4, 5, 6]])}")
    
    # Advanced exercises
    print(f"\nAdvanced Exercises:")
    lru_ops = [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3), ("get", 2)]
    print(f"1. LRU Cache: {exercises.lru_cache_simulation(2, lru_ops)}")
    
    json_data = {"a": {"b": 1, "c": {"d": 1}}, "e": 1}
    print(f"2. JSON paths for value 1: {exercises.json_path_finder(json_data, 1)}")


if __name__ == "__main__":
    run_practice_exercises()

