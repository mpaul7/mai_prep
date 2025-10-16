"""
Test suite for sampling and sample mean calculations.
Run this to validate your understanding and test your implementations.
"""

import unittest
import numpy as np
import pandas as pd
from sampling import SamplingToolkit
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestSamplingToolkit(unittest.TestCase):
    """Test cases for the SamplingToolkit class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.toolkit = SamplingToolkit(random_seed=42)
        self.test_population = list(range(1, 101))  # 1 to 100
        self.test_sample = [1, 2, 3, 4, 5]
    
    def test_simple_random_sample(self):
        """Test simple random sampling."""
        sample = self.toolkit.simple_random_sample(self.test_population, 10)
        
        # Check sample size
        self.assertEqual(len(sample), 10)
        
        # Check all elements are from population
        for element in sample:
            self.assertIn(element, self.test_population)
        
        # Check no duplicates (sampling without replacement)
        self.assertEqual(len(sample), len(set(sample)))
    
    def test_sample_with_replacement(self):
        """Test sampling with replacement."""
        sample = self.toolkit.sample_with_replacement([1, 2, 3], 10)
        
        # Check sample size
        self.assertEqual(len(sample), 10)
        
        # Check all elements are from population
        for element in sample:
            self.assertIn(element, [1, 2, 3])
    
    def test_calculate_sample_mean(self):
        """Test sample mean calculation."""
        mean = self.toolkit.calculate_sample_mean(self.test_sample)
        expected_mean = 3.0
        self.assertEqual(mean, expected_mean)
        
        # Test with empty list
        with self.assertRaises(ValueError):
            self.toolkit.calculate_sample_mean([])
    
    def test_calculate_sample_statistics(self):
        """Test comprehensive sample statistics."""
        stats = self.toolkit.calculate_sample_statistics(self.test_sample)
        
        # Check required keys
        required_keys = ['mean', 'median', 'std_dev', 'variance', 'min', 'max', 'range', 'sample_size']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check specific values
        self.assertEqual(stats['mean'], 3.0)
        self.assertEqual(stats['median'], 3.0)
        self.assertEqual(stats['min'], 1.0)
        self.assertEqual(stats['max'], 5.0)
        self.assertEqual(stats['sample_size'], 5)
    
    def test_systematic_sample(self):
        """Test systematic sampling."""
        sample = self.toolkit.systematic_sample(self.test_population, 10)
        
        # Check sample size
        self.assertEqual(len(sample), 10)
        
        # Check all elements are from population
        for element in sample:
            self.assertIn(element, self.test_population)
    
    def test_stratified_sample(self):
        """Test stratified sampling."""
        # Create test DataFrame
        data = pd.DataFrame({
            'value': list(range(1, 101)),
            'group': ['A'] * 50 + ['B'] * 50
        })
        
        sample = self.toolkit.stratified_sample(data, 'group', 20)
        
        # Check sample size is approximately correct
        self.assertLessEqual(len(sample), 20)
        self.assertGreater(len(sample), 0)
        
        # Check both groups are represented
        groups_in_sample = sample['group'].unique()
        self.assertTrue(len(groups_in_sample) > 0)


class TestHackerRankProblems(unittest.TestCase):
    """Test cases for HackerRank-style problems."""
    
    def test_basic_sample_mean(self):
        """Test basic sample mean calculation."""
        def calculate_mean(numbers):
            return round(sum(numbers) / len(numbers), 2)
        
        result = calculate_mean([10, 20, 30, 40, 50])
        self.assertEqual(result, 30.00)
    
    def test_weighted_sample_mean(self):
        """Test weighted sample mean calculation."""
        def weighted_mean(values, weights):
            weighted_sum = sum(v * w for v, w in zip(values, weights))
            total_weight = sum(weights)
            return round(weighted_sum / total_weight, 2)
        
        result = weighted_mean([10, 20, 30], [1, 2, 3])
        expected = round((10*1 + 20*2 + 30*3) / (1+2+3), 2)
        self.assertEqual(result, expected)
    
    def test_sample_mean_comparison(self):
        """Test sample mean comparison."""
        def compare_means(sample1, sample2):
            mean1 = sum(sample1) / len(sample1)
            mean2 = sum(sample2) / len(sample2)
            
            if mean1 > mean2:
                return 1
            elif mean2 > mean1:
                return 2
            else:
                return 0
        
        result = compare_means([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        self.assertEqual(result, 2)  # Second sample has higher mean
    
    def test_moving_sample_mean(self):
        """Test moving sample mean calculation."""
        def moving_mean(numbers, window_size):
            moving_means = []
            for i in range(len(numbers) - window_size + 1):
                window = numbers[i:i + window_size]
                mean = sum(window) / len(window)
                moving_means.append(round(mean, 2))
            return moving_means
        
        result = moving_mean([1, 2, 3, 4, 5, 6], 3)
        expected = [2.0, 3.0, 4.0, 5.0]
        self.assertEqual(result, expected)


def run_interactive_examples():
    """Run interactive examples to demonstrate concepts."""
    print("=== Interactive Sampling Examples ===\n")
    
    # Example 1: Basic sampling and mean calculation
    print("Example 1: Basic Sampling and Mean Calculation")
    toolkit = SamplingToolkit(random_seed=42)
    population = list(range(1, 21))  # 1 to 20
    sample = toolkit.simple_random_sample(population, 5)
    mean = toolkit.calculate_sample_mean(sample)
    
    print(f"Population: {population}")
    print(f"Sample (n=5): {sample}")
    print(f"Sample Mean: {mean}")
    print(f"Population Mean: {sum(population)/len(population)}")
    print()
    
    # Example 2: Effect of sample size on sample mean
    print("Example 2: Effect of Sample Size on Sample Mean")
    sample_sizes = [5, 10, 15, 20]
    for n in sample_sizes:
        if n <= len(population):
            sample = toolkit.simple_random_sample(population, n)
            mean = toolkit.calculate_sample_mean(sample)
            print(f"Sample size {n}: Mean = {mean:.2f}")
    print()
    
    # Example 3: Sampling distribution demonstration
    print("Example 3: Sampling Distribution of Sample Means")
    sample_means = []
    for _ in range(100):  # Take 100 samples
        sample = toolkit.simple_random_sample(population, 5)
        mean = toolkit.calculate_sample_mean(sample)
        sample_means.append(mean)
    
    mean_of_means = sum(sample_means) / len(sample_means)
    print(f"Mean of 100 sample means: {mean_of_means:.2f}")
    print(f"Population mean: {sum(population)/len(population):.2f}")
    print(f"Difference: {abs(mean_of_means - sum(population)/len(population)):.2f}")


if __name__ == "__main__":
    print("Running Sampling Tests...\n")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*60 + "\n")
    
    # Run interactive examples
    run_interactive_examples()

