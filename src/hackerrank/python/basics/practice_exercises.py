"""
Python Practice Exercises for Data Science Interview Preparation

This module contains additional practice exercises and challenges
commonly found in technical interviews and coding assessments.

Categories:
1. Basic Programming Challenges
2. String Manipulation
3. List and Array Operations
4. Mathematical Problems
5. Pattern Recognition
6. Data Processing
7. Algorithm Implementation
"""

from typing import List, Dict, Tuple, Union, Optional
import math


class PythonPracticeExercises:
    """
    Collection of Python practice exercises for interview preparation.
    """
    
    def __init__(self):
        """Initialize the practice exercises."""
        pass
    
    # ========== BASIC PROGRAMMING CHALLENGES ==========
    
    def reverse_string(self, s: str) -> str:
        """
        Reverse a string without using built-in reverse functions.
        
        Args:
            s: Input string
            
        Returns:
            Reversed string
            
        Example:
            >>> exercises = PythonPracticeExercises()
            >>> exercises.reverse_string("hello")
            'olleh'
        """
        result = ""
        for i in range(len(s) - 1, -1, -1):
            result += s[i]
        return result
    
    def is_palindrome(self, s: str) -> bool:
        """
        Check if a string is a palindrome (reads same forwards and backwards).
        
        Args:
            s: Input string
            
        Returns:
            True if palindrome, False otherwise
        """
        # Convert to lowercase and remove non-alphanumeric characters
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        
        left, right = 0, len(cleaned) - 1
        while left < right:
            if cleaned[left] != cleaned[right]:
                return False
            left += 1
            right -= 1
        
        return True
    
    def factorial(self, n: int) -> int:
        """
        Calculate factorial of a number using iteration.
        
        Args:
            n: Non-negative integer
            
        Returns:
            Factorial of n
        """
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        
        result = 1
        for i in range(1, n + 1):
            result *= i
        
        return result
    
    def fibonacci_sequence(self, n: int) -> List[int]:
        """
        Generate first n numbers in Fibonacci sequence.
        
        Args:
            n: Number of Fibonacci numbers to generate
            
        Returns:
            List of first n Fibonacci numbers
        """
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
        
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    # ========== STRING MANIPULATION ==========
    
    def count_vowels_consonants(self, s: str) -> Dict[str, int]:
        """
        Count vowels and consonants in a string.
        
        Args:
            s: Input string
            
        Returns:
            Dictionary with vowel and consonant counts
        """
        vowels = "aeiouAEIOU"
        vowel_count = 0
        consonant_count = 0
        
        for char in s:
            if char.isalpha():
                if char in vowels:
                    vowel_count += 1
                else:
                    consonant_count += 1
        
        return {
            'vowels': vowel_count,
            'consonants': consonant_count,
            'total_letters': vowel_count + consonant_count
        }
    
    def remove_duplicates_string(self, s: str) -> str:
        """
        Remove duplicate characters from string while preserving order.
        
        Args:
            s: Input string
            
        Returns:
            String with duplicates removed
        """
        seen = set()
        result = []
        
        for char in s:
            if char not in seen:
                seen.add(char)
                result.append(char)
        
        return ''.join(result)
    
    def word_frequency(self, text: str) -> Dict[str, int]:
        """
        Count frequency of each word in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with word frequencies
        """
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Remove punctuation from words
        cleaned_words = []
        for word in words:
            cleaned_word = ''.join(char for char in word if char.isalnum())
            if cleaned_word:
                cleaned_words.append(cleaned_word)
        
        # Count frequencies
        frequency = {}
        for word in cleaned_words:
            frequency[word] = frequency.get(word, 0) + 1
        
        return frequency
    
    def longest_common_prefix(self, strings: List[str]) -> str:
        """
        Find the longest common prefix among a list of strings.
        
        Args:
            strings: List of strings
            
        Returns:
            Longest common prefix
        """
        if not strings:
            return ""
        
        # Start with the first string as reference
        prefix = strings[0]
        
        for string in strings[1:]:
            # Reduce prefix until it matches the beginning of current string
            while not string.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        
        return prefix
    
    # ========== LIST AND ARRAY OPERATIONS ==========
    
    def find_duplicates(self, nums: List[int]) -> List[int]:
        """
        Find all duplicate numbers in a list.
        
        Args:
            nums: List of integers
            
        Returns:
            List of duplicate numbers
        """
        seen = set()
        duplicates = set()
        
        for num in nums:
            if num in seen:
                duplicates.add(num)
            else:
                seen.add(num)
        
        return list(duplicates)
    
    def merge_sorted_lists(self, list1: List[int], list2: List[int]) -> List[int]:
        """
        Merge two sorted lists into one sorted list.
        
        Args:
            list1: First sorted list
            list2: Second sorted list
            
        Returns:
            Merged sorted list
        """
        result = []
        i, j = 0, 0
        
        # Merge while both lists have elements
        while i < len(list1) and j < len(list2):
            if list1[i] <= list2[j]:
                result.append(list1[i])
                i += 1
            else:
                result.append(list2[j])
                j += 1
        
        # Add remaining elements
        while i < len(list1):
            result.append(list1[i])
            i += 1
        
        while j < len(list2):
            result.append(list2[j])
            j += 1
        
        return result
    
    def rotate_list(self, nums: List[int], k: int) -> List[int]:
        """
        Rotate list to the right by k steps.
        
        Args:
            nums: List of integers
            k: Number of steps to rotate
            
        Returns:
            Rotated list
        """
        if not nums or k == 0:
            return nums
        
        n = len(nums)
        k = k % n  # Handle k > n
        
        return nums[-k:] + nums[:-k]
    
    def find_missing_number(self, nums: List[int]) -> int:
        """
        Find the missing number in a sequence from 0 to n.
        
        Args:
            nums: List containing n distinct numbers from 0 to n with one missing
            
        Returns:
            The missing number
        """
        n = len(nums)
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(nums)
        return expected_sum - actual_sum
    
    # ========== MATHEMATICAL PROBLEMS ==========
    
    def is_prime(self, n: int) -> bool:
        """
        Check if a number is prime.
        
        Args:
            n: Integer to check
            
        Returns:
            True if prime, False otherwise
        """
        if n < 2:
            return False
        
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        
        return True
    
    def gcd(self, a: int, b: int) -> int:
        """
        Find Greatest Common Divisor using Euclidean algorithm.
        
        Args:
            a: First integer
            b: Second integer
            
        Returns:
            GCD of a and b
        """
        while b:
            a, b = b, a % b
        return abs(a)
    
    def lcm(self, a: int, b: int) -> int:
        """
        Find Least Common Multiple.
        
        Args:
            a: First integer
            b: Second integer
            
        Returns:
            LCM of a and b
        """
        return abs(a * b) // self.gcd(a, b)
    
    def sum_of_digits(self, n: int) -> int:
        """
        Calculate sum of digits in a number.
        
        Args:
            n: Integer
            
        Returns:
            Sum of digits
        """
        total = 0
        n = abs(n)  # Handle negative numbers
        
        while n > 0:
            total += n % 10
            n //= 10
        
        return total
    
    def is_perfect_square(self, n: int) -> bool:
        """
        Check if a number is a perfect square.
        
        Args:
            n: Integer to check
            
        Returns:
            True if perfect square, False otherwise
        """
        if n < 0:
            return False
        
        sqrt_n = int(math.sqrt(n))
        return sqrt_n * sqrt_n == n
    
    # ========== PATTERN RECOGNITION ==========
    
    def generate_pascal_triangle(self, n: int) -> List[List[int]]:
        """
        Generate first n rows of Pascal's triangle.
        
        Args:
            n: Number of rows
            
        Returns:
            List of lists representing Pascal's triangle
        """
        if n <= 0:
            return []
        
        triangle = [[1]]
        
        for i in range(1, n):
            prev_row = triangle[i-1]
            new_row = [1]
            
            for j in range(1, i):
                new_row.append(prev_row[j-1] + prev_row[j])
            
            new_row.append(1)
            triangle.append(new_row)
        
        return triangle
    
    def print_diamond_pattern(self, n: int) -> List[str]:
        """
        Generate diamond pattern with stars.
        
        Args:
            n: Size of diamond (should be odd)
            
        Returns:
            List of strings representing diamond pattern
        """
        if n % 2 == 0:
            n += 1  # Make it odd
        
        pattern = []
        mid = n // 2
        
        # Upper half (including middle)
        for i in range(mid + 1):
            spaces = ' ' * (mid - i)
            stars = '*' * (2 * i + 1)
            pattern.append(spaces + stars)
        
        # Lower half
        for i in range(mid - 1, -1, -1):
            spaces = ' ' * (mid - i)
            stars = '*' * (2 * i + 1)
            pattern.append(spaces + stars)
        
        return pattern
    
    # ========== DATA PROCESSING ==========
    
    def group_by_key(self, data: List[Dict], key: str) -> Dict[str, List[Dict]]:
        """
        Group list of dictionaries by a specific key.
        
        Args:
            data: List of dictionaries
            key: Key to group by
            
        Returns:
            Dictionary with grouped data
        """
        grouped = {}
        
        for item in data:
            if key in item:
                group_key = item[key]
                if group_key not in grouped:
                    grouped[group_key] = []
                grouped[group_key].append(item)
        
        return grouped
    
    def calculate_statistics(self, numbers: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of numbers.
        
        Args:
            numbers: List of numbers
            
        Returns:
            Dictionary with statistics
        """
        if not numbers:
            return {}
        
        sorted_nums = sorted(numbers)
        n = len(numbers)
        
        # Mean
        mean = sum(numbers) / n
        
        # Median
        if n % 2 == 0:
            median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
        else:
            median = sorted_nums[n//2]
        
        # Mode (most frequent number)
        frequency = {}
        for num in numbers:
            frequency[num] = frequency.get(num, 0) + 1
        
        max_freq = max(frequency.values())
        modes = [num for num, freq in frequency.items() if freq == max_freq]
        mode = modes[0] if len(modes) == 1 else None
        
        # Range
        range_val = max(numbers) - min(numbers)
        
        # Variance and Standard Deviation
        variance = sum((x - mean) ** 2 for x in numbers) / n
        std_dev = math.sqrt(variance)
        
        return {
            'count': n,
            'mean': round(mean, 4),
            'median': median,
            'mode': mode,
            'min': min(numbers),
            'max': max(numbers),
            'range': range_val,
            'variance': round(variance, 4),
            'std_dev': round(std_dev, 4)
        }
    
    def flatten_nested_list(self, nested_list: List) -> List:
        """
        Flatten a nested list structure.
        
        Args:
            nested_list: List that may contain nested lists
            
        Returns:
            Flattened list
        """
        result = []
        
        for item in nested_list:
            if isinstance(item, list):
                result.extend(self.flatten_nested_list(item))
            else:
                result.append(item)
        
        return result


def demonstrate_exercises():
    """Demonstrate the practice exercises with examples."""
    print("=== Python Practice Exercises Demonstration ===\n")
    
    exercises = PythonPracticeExercises()
    
    # Basic Programming Challenges
    print("1. Basic Programming Challenges:")
    print(f"   reverse_string('hello') = '{exercises.reverse_string('hello')}'")
    print(f"   is_palindrome('racecar') = {exercises.is_palindrome('racecar')}")
    print(f"   factorial(5) = {exercises.factorial(5)}")
    print(f"   fibonacci_sequence(8) = {exercises.fibonacci_sequence(8)}")
    print()
    
    # String Manipulation
    print("2. String Manipulation:")
    vowel_count = exercises.count_vowels_consonants("Hello World")
    print(f"   count_vowels_consonants('Hello World') = {vowel_count}")
    print(f"   remove_duplicates_string('programming') = '{exercises.remove_duplicates_string('programming')}'")
    
    word_freq = exercises.word_frequency("the quick brown fox jumps over the lazy dog")
    print(f"   word_frequency sample: {dict(list(word_freq.items())[:5])}")
    
    common_prefix = exercises.longest_common_prefix(["flower", "flow", "flight"])
    print(f"   longest_common_prefix(['flower', 'flow', 'flight']) = '{common_prefix}'")
    print()
    
    # List Operations
    print("3. List and Array Operations:")
    print(f"   find_duplicates([1,2,3,2,4,3,5]) = {exercises.find_duplicates([1,2,3,2,4,3,5])}")
    
    merged = exercises.merge_sorted_lists([1,3,5], [2,4,6])
    print(f"   merge_sorted_lists([1,3,5], [2,4,6]) = {merged}")
    
    rotated = exercises.rotate_list([1,2,3,4,5], 2)
    print(f"   rotate_list([1,2,3,4,5], 2) = {rotated}")
    
    missing = exercises.find_missing_number([0,1,3,4,5])
    print(f"   find_missing_number([0,1,3,4,5]) = {missing}")
    print()
    
    # Mathematical Problems
    print("4. Mathematical Problems:")
    print(f"   is_prime(17) = {exercises.is_prime(17)}")
    print(f"   gcd(48, 18) = {exercises.gcd(48, 18)}")
    print(f"   lcm(12, 8) = {exercises.lcm(12, 8)}")
    print(f"   sum_of_digits(12345) = {exercises.sum_of_digits(12345)}")
    print(f"   is_perfect_square(16) = {exercises.is_perfect_square(16)}")
    print()
    
    # Pattern Recognition
    print("5. Pattern Recognition:")
    pascal = exercises.generate_pascal_triangle(5)
    print("   Pascal's Triangle (5 rows):")
    for row in pascal:
        print(f"     {row}")
    
    print("   Diamond Pattern (size 5):")
    diamond = exercises.print_diamond_pattern(5)
    for line in diamond:
        print(f"     {line}")
    print()
    
    # Data Processing
    print("6. Data Processing:")
    sample_data = [
        {'name': 'Alice', 'department': 'Engineering', 'salary': 70000},
        {'name': 'Bob', 'department': 'Marketing', 'salary': 60000},
        {'name': 'Charlie', 'department': 'Engineering', 'salary': 75000}
    ]
    
    grouped = exercises.group_by_key(sample_data, 'department')
    print(f"   Grouped by department: {list(grouped.keys())}")
    
    stats = exercises.calculate_statistics([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
    print(f"   Statistics for [1,2,3,4,5,5,6,7,8,9]: mean={stats['mean']}, median={stats['median']}")
    
    nested = [1, [2, 3], [4, [5, 6]], 7]
    flattened = exercises.flatten_nested_list(nested)
    print(f"   flatten_nested_list({nested}) = {flattened}")


if __name__ == "__main__":
    demonstrate_exercises()
