"""
Sampling and Sample Mean Calculations for Data Science Interview Preparation

This module covers fundamental sampling concepts commonly tested in data science interviews,
particularly HackerRank-style problems.

Topics covered:
1. Simple Random Sampling
2. Sample Mean Calculation
3. Different sampling methods
4. Statistical properties of samples
"""

import numpy as np
import pandas as pd
import random
from typing import List, Union, Optional
import statistics
from collections import Counter


class SamplingToolkit:
    """
    A comprehensive toolkit for sampling operations and sample mean calculations.
    Designed for data science interview preparation.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the sampling toolkit.
        
        Args:
            random_seed: Optional seed for reproducible results
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def simple_random_sample(self, population: List[Union[int, float]], 
                           sample_size: int) -> List[Union[int, float]]:
        """
        Draw a simple random sample from a population.
        
        Args:
            population: The population to sample from
            sample_size: Number of elements to sample
            
        Returns:
            List of sampled elements
            
        Example:
            >>> toolkit = SamplingToolkit(42)
            >>> population = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> sample = toolkit.simple_random_sample(population, 5)
            >>> print(sample)  # [6, 9, 4, 8, 1]
        """
        if sample_size > len(population):
            raise ValueError("Sample size cannot be larger than population size")
        
        return random.sample(population, sample_size)
    
    def sample_with_replacement(self, population: List[Union[int, float]], 
                              sample_size: int) -> List[Union[int, float]]:
        """
        Draw a sample with replacement from a population.
        
        Args:
            population: The population to sample from
            sample_size: Number of elements to sample
            
        Returns:
            List of sampled elements (may contain duplicates)
        """
        return random.choices(population, k=sample_size)
    
    def calculate_sample_mean(self, sample: List[Union[int, float]]) -> float:
        """
        Calculate the sample mean.
        
        Args:
            sample: List of numeric values
            
        Returns:
            Sample mean as float
            
        Example:
            >>> toolkit = SamplingToolkit()
            >>> sample = [1, 2, 3, 4, 5]
            >>> mean = toolkit.calculate_sample_mean(sample)
            >>> print(mean)  # 3.0
        """
        if not sample:
            raise ValueError("Cannot calculate mean of empty sample")
        
        return sum(sample) / len(sample)
    
    def calculate_sample_statistics(self, sample: List[Union[int, float]]) -> dict:
        """
        Calculate comprehensive sample statistics.
        
        Args:
            sample: List of numeric values
            
        Returns:
            Dictionary containing various sample statistics
        """
        if not sample:
            raise ValueError("Cannot calculate statistics of empty sample")
        
        sample_array = np.array(sample)
        
        return {
            'mean': float(np.mean(sample_array)),
            'median': float(np.median(sample_array)),
            'mode': statistics.mode(sample) if len(set(sample)) < len(sample) else None,
            'std_dev': float(np.std(sample_array, ddof=1)),  # Sample standard deviation
            'variance': float(np.var(sample_array, ddof=1)),  # Sample variance
            'min': float(np.min(sample_array)),
            'max': float(np.max(sample_array)),
            'range': float(np.max(sample_array) - np.min(sample_array)),
            'sample_size': len(sample)
        }
    
    def stratified_sample(self, data: pd.DataFrame, strata_column: str, 
                         sample_size: int) -> pd.DataFrame:
        """
        Perform stratified sampling on a DataFrame.
        
        Args:
            data: DataFrame to sample from
            strata_column: Column name to use for stratification
            sample_size: Total sample size
            
        Returns:
            Stratified sample as DataFrame
        """
        strata_counts = data[strata_column].value_counts()
        strata_proportions = strata_counts / len(data)
        
        sampled_data = []
        
        for stratum, proportion in strata_proportions.items():
            stratum_sample_size = int(sample_size * proportion)
            if stratum_sample_size > 0:
                stratum_data = data[data[strata_column] == stratum]
                stratum_sample = stratum_data.sample(n=min(stratum_sample_size, len(stratum_data)))
                sampled_data.append(stratum_sample)
        
        return pd.concat(sampled_data, ignore_index=True)
    
    def systematic_sample(self, population: List[Union[int, float]], 
                         sample_size: int) -> List[Union[int, float]]:
        """
        Perform systematic sampling.
        
        Args:
            population: The population to sample from
            sample_size: Number of elements to sample
            
        Returns:
            Systematically sampled elements
        """
        if sample_size > len(population):
            raise ValueError("Sample size cannot be larger than population size")
        
        interval = len(population) // sample_size
        start = random.randint(0, interval - 1)
        
        sample = []
        for i in range(sample_size):
            index = (start + i * interval) % len(population)
            sample.append(population[index])
        
        return sample


def hackerrank_style_problems():
    """
    Collection of HackerRank-style problems for sampling and sample means.
    """
    
    def problem_1_basic_sample_mean():
        """
        Problem 1: Calculate Sample Mean
        
        Given a list of numbers, calculate the sample mean.
        Round to 2 decimal places.
        
        Input: [10, 20, 30, 40, 50]
        Expected Output: 30.00
        """
        def solution(numbers: List[Union[int, float]]) -> float:
            return round(sum(numbers) / len(numbers), 2)
        
        # Test case
        test_input = [10, 20, 30, 40, 50]
        result = solution(test_input)
        print(f"Problem 1 - Input: {test_input}")
        print(f"Problem 1 - Sample Mean: {result}")
        return result
    
    def problem_2_weighted_sample_mean():
        """
        Problem 2: Weighted Sample Mean
        
        Given a list of values and their corresponding weights,
        calculate the weighted sample mean.
        
        Input: values = [10, 20, 30], weights = [1, 2, 3]
        Expected Output: 23.33
        """
        def solution(values: List[Union[int, float]], 
                    weights: List[Union[int, float]]) -> float:
            if len(values) != len(weights):
                raise ValueError("Values and weights must have same length")
            
            weighted_sum = sum(v * w for v, w in zip(values, weights))
            total_weight = sum(weights)
            return round(weighted_sum / total_weight, 2)
        
        # Test case
        test_values = [10, 20, 30]
        test_weights = [1, 2, 3]
        result = solution(test_values, test_weights)
        print(f"Problem 2 - Values: {test_values}, Weights: {test_weights}")
        print(f"Problem 2 - Weighted Sample Mean: {result}")
        return result
    
    def problem_3_sample_mean_comparison():
        """
        Problem 3: Compare Sample Means
        
        Given two samples, determine which has a higher sample mean.
        Return 1 if first sample has higher mean, 2 if second sample has higher mean,
        0 if they are equal.
        
        Input: sample1 = [1, 2, 3, 4, 5], sample2 = [2, 3, 4, 5, 6]
        Expected Output: 2
        """
        def solution(sample1: List[Union[int, float]], 
                    sample2: List[Union[int, float]]) -> int:
            mean1 = sum(sample1) / len(sample1)
            mean2 = sum(sample2) / len(sample2)
            
            if mean1 > mean2:
                return 1
            elif mean2 > mean1:
                return 2
            else:
                return 0
        
        # Test case
        test_sample1 = [1, 2, 3, 4, 5]
        test_sample2 = [2, 3, 4, 5, 6]
        result = solution(test_sample1, test_sample2)
        print(f"Problem 3 - Sample 1: {test_sample1} (mean: {sum(test_sample1)/len(test_sample1)})")
        print(f"Problem 3 - Sample 2: {test_sample2} (mean: {sum(test_sample2)/len(test_sample2)})")
        print(f"Problem 3 - Result: {result}")
        return result
    
    def problem_4_moving_sample_mean():
        """
        Problem 4: Moving Sample Mean
        
        Given a list of numbers and a window size, calculate the moving sample mean.
        
        Input: numbers = [1, 2, 3, 4, 5, 6], window_size = 3
        Expected Output: [2.0, 3.0, 4.0, 5.0]
        """
        def solution(numbers: List[Union[int, float]], window_size: int) -> List[float]:
            if window_size > len(numbers):
                raise ValueError("Window size cannot be larger than list length")
            
            moving_means = []
            for i in range(len(numbers) - window_size + 1):
                window = numbers[i:i + window_size]
                mean = sum(window) / len(window)
                moving_means.append(round(mean, 2))
            
            return moving_means
        
        # Test case
        test_numbers = [1, 2, 3, 4, 5, 6]
        test_window = 3
        result = solution(test_numbers, test_window)
        print(f"Problem 4 - Numbers: {test_numbers}, Window Size: {test_window}")
        print(f"Problem 4 - Moving Sample Means: {result}")
        return result
    
    # Run all problems
    print("=== HackerRank Style Sampling Problems ===\n")
    problem_1_basic_sample_mean()
    print()
    problem_2_weighted_sample_mean()
    print()
    problem_3_sample_mean_comparison()
    print()
    problem_4_moving_sample_mean()


if __name__ == "__main__":
    # Demonstrate the sampling toolkit
    print("=== Sampling Toolkit Demonstration ===\n")
    
    # Initialize toolkit with seed for reproducible results
    toolkit = SamplingToolkit(random_seed=42)
    
    # Example population
    population = list(range(1, 101))  # Numbers 1 to 100
    print(f"Population: {population[:10]}... (size: {len(population)})")
    
    # Simple random sample
    sample = toolkit.simple_random_sample(population, 10)
    print(f"Simple Random Sample (n=10): {sample}")
    
    # Calculate sample mean
    sample_mean = toolkit.calculate_sample_mean(sample)
    print(f"Sample Mean: {sample_mean:.2f}")
    
    # Comprehensive statistics
    stats = toolkit.calculate_sample_statistics(sample)
    print(f"Sample Statistics: {stats}")
    
    print("\n" + "="*50 + "\n")
    
    # Run HackerRank-style problems
    hackerrank_style_problems()
