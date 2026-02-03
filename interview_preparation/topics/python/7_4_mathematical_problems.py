"""
Python Mathematical Problems - Interview Preparation
Topic 7.4: Mathematical Problems

This module covers:
- Number Operations: Prime checking, factorial, GCD/LCM
- Digit Manipulation: Extracting digits, reversing numbers
- Mathematical Formulas: Implementing formulas
"""

import math
from functools import lru_cache
from typing import List

# ============================================================================
# 1. NUMBER OPERATIONS
# ============================================================================

print("=" * 70)
print("1. NUMBER OPERATIONS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Prime Checking
# ----------------------------------------------------------------------------
print("\n--- 1.1 Prime Checking ---")

def is_prime_basic(n):
    """
    Check if number is prime - basic approach.
    Time: O(√n)
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to √n
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    
    return True

def is_prime_optimized(n):
    """
    Optimized prime check with early exits.
    Time: O(√n)
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Check divisors of form 6k±1
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True

def sieve_of_eratosthenes(n):
    """
    Generate all primes up to n using Sieve of Eratosthenes.
    Time: O(n log log n), Space: O(n)
    """
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            # Mark multiples of i as not prime
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]

# Test prime checking
test_numbers = [2, 3, 4, 17, 25, 29, 97]
print("Prime checking:")
for num in test_numbers:
    result = is_prime_basic(num)
    print(f"  {num}: {result}")

# Generate primes
primes = sieve_of_eratosthenes(30)
print(f"\nPrimes up to 30: {primes}")


# ----------------------------------------------------------------------------
# 1.2 Factorial
# ----------------------------------------------------------------------------
print("\n--- 1.2 Factorial ---")

def factorial_iterative(n):
    """
    Calculate factorial iteratively.
    Time: O(n), Space: O(1)
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def factorial_recursive(n):
    """
    Calculate factorial recursively.
    Time: O(n), Space: O(n) due to call stack
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)

@lru_cache(maxsize=128)
def factorial_cached(n):
    """Factorial with caching."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial_cached(n - 1)

def factorial_math(n):
    """Factorial using math module."""
    return math.factorial(n)

print("Factorial calculations:")
for n in [0, 1, 5, 10]:
    print(f"  {n}! = {factorial_iterative(n)}")
    print(f"  {n}! (recursive) = {factorial_recursive(n)}")
    print(f"  {n}! (math) = {factorial_math(n)}")


# ----------------------------------------------------------------------------
# 1.3 GCD (Greatest Common Divisor)
# ----------------------------------------------------------------------------
print("\n--- 1.3 GCD (Greatest Common Divisor) ---")

def gcd_euclidean(a, b):
    """
    Calculate GCD using Euclidean algorithm (iterative).
    Time: O(log min(a, b))
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def gcd_euclidean_recursive(a, b):
    """GCD using Euclidean algorithm (recursive)."""
    a, b = abs(a), abs(b)
    if b == 0:
        return a
    return gcd_euclidean_recursive(b, a % b)

def gcd_math(a, b):
    """GCD using math module."""
    return math.gcd(a, b)

def gcd_multiple(numbers):
    """Calculate GCD of multiple numbers."""
    if not numbers:
        return 0
    result = numbers[0]
    for num in numbers[1:]:
        result = gcd_euclidean(result, num)
    return result

print("GCD calculations:")
print(f"  gcd(48, 18) = {gcd_euclidean(48, 18)}")
print(f"  gcd(100, 25) = {gcd_euclidean(100, 25)}")
print(f"  gcd(17, 13) = {gcd_euclidean(17, 13)}")
print(f"  gcd([48, 18, 12]) = {gcd_multiple([48, 18, 12])}")


# ----------------------------------------------------------------------------
# 1.4 LCM (Least Common Multiple)
# ----------------------------------------------------------------------------
print("\n--- 1.4 LCM (Least Common Multiple) ---")

def lcm(a, b):
    """
    Calculate LCM using formula: LCM(a, b) = |a * b| / GCD(a, b)
    Time: O(log min(a, b))
    """
    return abs(a * b) // gcd_euclidean(a, b)

def lcm_math(a, b):
    """LCM using math module (Python 3.9+)."""
    return math.lcm(a, b)

def lcm_multiple(numbers):
    """Calculate LCM of multiple numbers."""
    if not numbers:
        return 0
    result = numbers[0]
    for num in numbers[1:]:
        result = lcm(result, num)
    return result

print("LCM calculations:")
print(f"  lcm(12, 8) = {lcm(12, 8)}")
print(f"  lcm(15, 20) = {lcm(15, 20)}")
print(f"  lcm([12, 8, 6]) = {lcm_multiple([12, 8, 6])}")


# ----------------------------------------------------------------------------
# 1.5 Power and Exponentiation
# ----------------------------------------------------------------------------
print("\n--- 1.5 Power and Exponentiation ---")

def power_iterative(base, exponent):
    """
    Calculate base^exponent iteratively.
    Time: O(exponent)
    """
    if exponent < 0:
        return 1 / power_iterative(base, -exponent)
    if exponent == 0:
        return 1
    
    result = 1
    for _ in range(exponent):
        result *= base
    return result

def power_recursive(base, exponent):
    """
    Calculate base^exponent recursively.
    Time: O(exponent)
    """
    if exponent < 0:
        return 1 / power_recursive(base, -exponent)
    if exponent == 0:
        return 1
    return base * power_recursive(base, exponent - 1)

def power_optimized(base, exponent):
    """
    Calculate base^exponent using fast exponentiation.
    Time: O(log exponent)
    """
    if exponent < 0:
        return 1 / power_optimized(base, -exponent)
    if exponent == 0:
        return 1
    
    result = 1
    current_power = base
    
    while exponent > 0:
        if exponent % 2 == 1:
            result *= current_power
        current_power *= current_power
        exponent //= 2
    
    return result

print("Power calculations:")
print(f"  2^10 (iterative) = {power_iterative(2, 10)}")
print(f"  2^10 (recursive) = {power_recursive(2, 10)}")
print(f"  2^10 (optimized) = {power_optimized(2, 10)}")
print(f"  2^10 (built-in) = {2 ** 10}")


# ============================================================================
# 2. DIGIT MANIPULATION
# ============================================================================

print("\n" + "=" * 70)
print("2. DIGIT MANIPULATION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Extracting Digits
# ----------------------------------------------------------------------------
print("\n--- 2.1 Extracting Digits ---")

def extract_digits(n):
    """
    Extract all digits from a number.
    Returns list of digits (most significant first).
    """
    if n == 0:
        return [0]
    
    digits = []
    n = abs(n)  # Handle negative numbers
    
    while n > 0:
        digits.append(n % 10)
        n //= 10
    
    return digits[::-1]  # Reverse to get most significant first

def extract_digits_string(n):
    """Extract digits using string conversion."""
    return [int(d) for d in str(abs(n))]

number = 12345
print(f"Digits of {number}:")
print(f"  Method 1: {extract_digits(number)}")
print(f"  Method 2: {extract_digits_string(number)}")


# ----------------------------------------------------------------------------
# 2.2 Reversing Numbers
# ----------------------------------------------------------------------------
print("\n--- 2.2 Reversing Numbers ---")

def reverse_number(n):
    """
    Reverse digits of a number.
    Example: 12345 -> 54321
    """
    reversed_num = 0
    n_abs = abs(n)
    sign = 1 if n >= 0 else -1
    
    while n_abs > 0:
        reversed_num = reversed_num * 10 + n_abs % 10
        n_abs //= 10
    
    return sign * reversed_num

def reverse_number_string(n):
    """Reverse number using string conversion."""
    sign = -1 if n < 0 else 1
    return sign * int(str(abs(n))[::-1])

numbers = [12345, -12345, 100, 0]
print("Reversing numbers:")
for n in numbers:
    print(f"  {n} -> {reverse_number(n)}")


# ----------------------------------------------------------------------------
# 2.3 Sum of Digits
# ----------------------------------------------------------------------------
print("\n--- 2.3 Sum of Digits ---")

def sum_of_digits(n):
    """Calculate sum of all digits."""
    total = 0
    n = abs(n)
    
    while n > 0:
        total += n % 10
        n //= 10
    
    return total

def sum_of_digits_string(n):
    """Sum of digits using string conversion."""
    return sum(int(d) for d in str(abs(n)))

def sum_of_digits_recursive(n):
    """Sum of digits recursively."""
    n = abs(n)
    if n == 0:
        return 0
    return (n % 10) + sum_of_digits_recursive(n // 10)

number = 12345
print(f"Sum of digits of {number}:")
print(f"  Method 1: {sum_of_digits(number)}")
print(f"  Method 2: {sum_of_digits_string(number)}")
print(f"  Method 3: {sum_of_digits_recursive(number)}")


# ----------------------------------------------------------------------------
# 2.4 Product of Digits
# ----------------------------------------------------------------------------
print("\n--- 2.4 Product of Digits ---")

def product_of_digits(n):
    """Calculate product of all digits."""
    if n == 0:
        return 0
    
    product = 1
    n = abs(n)
    
    while n > 0:
        product *= n % 10
        n //= 10
    
    return product

number = 1234
print(f"Product of digits of {number}: {product_of_digits(number)}")


# ----------------------------------------------------------------------------
# 2.5 Count Digits
# ----------------------------------------------------------------------------
print("\n--- 2.5 Count Digits ---")

def count_digits(n):
    """Count number of digits."""
    if n == 0:
        return 1
    
    count = 0
    n = abs(n)
    
    while n > 0:
        count += 1
        n //= 10
    
    return count

def count_digits_log(n):
    """Count digits using logarithm."""
    if n == 0:
        return 1
    return int(math.log10(abs(n))) + 1

def count_digits_string(n):
    """Count digits using string conversion."""
    return len(str(abs(n)))

number = 12345
print(f"Number of digits in {number}:")
print(f"  Method 1: {count_digits(number)}")
print(f"  Method 2: {count_digits_log(number)}")
print(f"  Method 3: {count_digits_string(number)}")


# ----------------------------------------------------------------------------
# 2.6 Check if Number is Palindrome
# ----------------------------------------------------------------------------
print("\n--- 2.6 Check if Number is Palindrome ---")

def is_number_palindrome(n):
    """Check if number reads same forwards and backwards."""
    if n < 0:
        return False
    
    original = n
    reversed_num = 0
    
    while n > 0:
        reversed_num = reversed_num * 10 + n % 10
        n //= 10
    
    return original == reversed_num

numbers = [121, 12321, 123, -121, 0]
print("Number palindrome check:")
for n in numbers:
    print(f"  {n}: {is_number_palindrome(n)}")


# ----------------------------------------------------------------------------
# 2.7 Armstrong Number
# ----------------------------------------------------------------------------
print("\n--- 2.7 Armstrong Number ---")

def is_armstrong_number(n):
    """
    Check if number is Armstrong number.
    Armstrong: sum of digits raised to power of number of digits equals number.
    Example: 153 = 1^3 + 5^3 + 3^3
    """
    if n < 0:
        return False
    
    digits = extract_digits(n)
    num_digits = len(digits)
    total = sum(d ** num_digits for d in digits)
    
    return total == n

numbers = [153, 371, 9474, 123]
print("Armstrong number check:")
for n in numbers:
    print(f"  {n}: {is_armstrong_number(n)}")


# ============================================================================
# 3. MATHEMATICAL FORMULAS
# ============================================================================

print("\n" + "=" * 70)
print("3. MATHEMATICAL FORMULAS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Fibonacci Sequence
# ----------------------------------------------------------------------------
print("\n--- 3.1 Fibonacci Sequence ---")

def fibonacci_iterative(n):
    """
    Generate Fibonacci sequence iteratively.
    Time: O(n), Space: O(1)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    
    return fib

def fibonacci_nth(n):
    """Get nth Fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

@lru_cache(maxsize=128)
def fibonacci_recursive(n):
    """Fibonacci recursively with caching."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

print("Fibonacci sequence:")
print(f"  First 10: {fibonacci_iterative(10)}")
print(f"  10th Fibonacci: {fibonacci_nth(10)}")
print(f"  10th Fibonacci (recursive): {fibonacci_recursive(10)}")


# ----------------------------------------------------------------------------
# 3.2 Perfect Number
# ----------------------------------------------------------------------------
print("\n--- 3.2 Perfect Number ---")

def is_perfect_number(n):
    """
    Check if number is perfect.
    Perfect number: sum of proper divisors equals the number.
    Example: 6 = 1 + 2 + 3
    """
    if n < 1:
        return False
    
    divisors_sum = 1  # 1 is always a divisor
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors_sum += i
            if i != n // i:  # Avoid counting square root twice
                divisors_sum += n // i
    
    return divisors_sum == n

numbers = [6, 28, 496, 12]
print("Perfect number check:")
for n in numbers:
    print(f"  {n}: {is_perfect_number(n)}")


# ----------------------------------------------------------------------------
# 3.3 Square Root (Integer)
# ----------------------------------------------------------------------------
print("\n--- 3.3 Square Root (Integer) ---")

def integer_sqrt(n):
    """
    Find integer square root using binary search.
    Returns largest integer whose square <= n
    """
    if n < 0:
        raise ValueError("Square root not defined for negative numbers")
    if n <= 1:
        return n
    
    left, right = 0, n
    result = 0
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == n:
            return mid
        elif square < n:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def is_perfect_square(n):
    """Check if number is perfect square."""
    if n < 0:
        return False
    root = integer_sqrt(n)
    return root * root == n

numbers = [16, 25, 30, 100]
print("Square root and perfect square:")
for n in numbers:
    print(f"  sqrt({n}) = {integer_sqrt(n)}, is perfect square: {is_perfect_square(n)}")


# ----------------------------------------------------------------------------
# 3.4 Sum of Arithmetic Series
# ----------------------------------------------------------------------------
print("\n--- 3.4 Sum of Arithmetic Series ---")

def sum_arithmetic_series(first, last, n):
    """
    Sum of arithmetic series: n/2 * (first + last)
    """
    return n * (first + last) // 2

def sum_1_to_n(n):
    """Sum of 1 to n."""
    return n * (n + 1) // 2

print(f"Sum 1 to 10: {sum_1_to_n(10)}")
print(f"Sum 1 to 100: {sum_1_to_n(100)}")


# ----------------------------------------------------------------------------
# 3.5 Sum of Geometric Series
# ----------------------------------------------------------------------------
print("\n--- 3.5 Sum of Geometric Series ---")

def sum_geometric_series(a, r, n):
    """
    Sum of geometric series: a * (r^n - 1) / (r - 1)
    a = first term, r = common ratio, n = number of terms
    """
    if r == 1:
        return a * n
    return a * (r ** n - 1) // (r - 1)

print(f"Geometric series (a=1, r=2, n=5): {sum_geometric_series(1, 2, 5)}")


# ----------------------------------------------------------------------------
# 3.6 Binomial Coefficient
# ----------------------------------------------------------------------------
print("\n--- 3.6 Binomial Coefficient ---")

def binomial_coefficient(n, k):
    """
    Calculate C(n, k) = n! / (k! * (n-k)!)
    """
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use symmetry: C(n, k) = C(n, n-k)
    if k > n - k:
        k = n - k
    
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    
    return result

def binomial_coefficient_math(n, k):
    """Binomial coefficient using math module."""
    return math.comb(n, k)

print("Binomial coefficients:")
print(f"  C(5, 2) = {binomial_coefficient(5, 2)}")
print(f"  C(10, 3) = {binomial_coefficient(10, 3)}")


# ----------------------------------------------------------------------------
# 3.7 Pascal's Triangle
# ----------------------------------------------------------------------------
print("\n--- 3.7 Pascal's Triangle ---")

def pascals_triangle(n):
    """
    Generate Pascal's triangle up to n rows.
    """
    triangle = []
    for i in range(n):
        row = [binomial_coefficient(i, j) for j in range(i + 1)]
        triangle.append(row)
    return triangle

def print_pascals_triangle(n):
    """Print Pascal's triangle nicely."""
    triangle = pascals_triangle(n)
    for row in triangle:
        print(' '.join(str(x) for x in row).center(50))

print("Pascal's triangle (5 rows):")
print_pascals_triangle(5)


# ============================================================================
# 4. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("4. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Check if Prime
print("\n--- Exercise 1: Check if Prime ---")
def check_prime(n):
    """Check if number is prime."""
    return is_prime_basic(n)

test_primes = [2, 17, 25, 29, 97]
for n in test_primes:
    print(f"  {n} is prime: {check_prime(n)}")


# Exercise 2: Factorial
print("\n--- Exercise 2: Factorial ---")
def calculate_factorial(n):
    """Calculate factorial."""
    return factorial_iterative(n)

for n in [0, 5, 10]:
    print(f"  {n}! = {calculate_factorial(n)}")


# Exercise 3: GCD and LCM
print("\n--- Exercise 3: GCD and LCM ---")
def calculate_gcd_lcm(a, b):
    """Calculate GCD and LCM."""
    gcd_val = gcd_euclidean(a, b)
    lcm_val = lcm(a, b)
    return gcd_val, lcm_val

a, b = 48, 18
gcd_val, lcm_val = calculate_gcd_lcm(a, b)
print(f"  GCD({a}, {b}) = {gcd_val}")
print(f"  LCM({a}, {b}) = {lcm_val}")


# Exercise 4: Sum of Digits
print("\n--- Exercise 4: Sum of Digits ---")
def digit_sum(n):
    """Calculate sum of digits."""
    return sum_of_digits(n)

numbers = [12345, 9876, 100]
for n in numbers:
    print(f"  Sum of digits of {n}: {digit_sum(n)}")


# Exercise 5: Reverse Number
print("\n--- Exercise 5: Reverse Number ---")
def reverse_num(n):
    """Reverse a number."""
    return reverse_number(n)

numbers = [12345, -12345, 100]
for n in numbers:
    print(f"  Reverse of {n}: {reverse_num(n)}")


# Exercise 6: Check Perfect Square
print("\n--- Exercise 6: Check Perfect Square ---")
def check_perfect_square(n):
    """Check if number is perfect square."""
    return is_perfect_square(n)

numbers = [16, 25, 30, 100]
for n in numbers:
    print(f"  {n} is perfect square: {check_perfect_square(n)}")


# ============================================================================
# 5. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("5. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. PRIME CHECKING:
   - Check up to √n (not n)
   - Skip even numbers after 2
   - Use Sieve for multiple primes
   - Time: O(√n) for single, O(n log log n) for sieve

2. FACTORIAL:
   - Iterative: O(n) time, O(1) space
   - Recursive: O(n) time, O(n) space
   - Use math.factorial() for built-in
   - Cache for repeated calculations

3. GCD/LCM:
   - Euclidean algorithm: O(log min(a, b))
   - LCM = |a * b| / GCD(a, b)
   - Use math.gcd() and math.lcm() (Python 3.9+)
   - Works for multiple numbers iteratively

4. DIGIT MANIPULATION:
   - Extract: n % 10, n // 10
   - Reverse: build number digit by digit
   - Count: use log10 or string conversion
   - Handle negative numbers

5. MATHEMATICAL FORMULAS:
   - Fibonacci: iterative O(n) or matrix O(log n)
   - Power: fast exponentiation O(log n)
   - Series: use formulas when possible
   - Binomial: use iterative calculation

6. BEST PRACTICES:
   - Handle edge cases (0, negative, 1)
   - Consider overflow for large numbers
   - Use built-in functions when available
   - Optimize with mathematical insights
   - Cache expensive calculations
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Mathematical Problems Guide Ready!")
    print("=" * 70)
