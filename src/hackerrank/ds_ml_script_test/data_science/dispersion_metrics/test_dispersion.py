"""
Test suite for dispersion metrics calculations.
Run this to validate your understanding and test your implementations.
"""

import unittest
import numpy as np
import pandas as pd
from dispersion import DispersionCalculator
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestDispersionCalculator(unittest.TestCase):
    """Test cases for the DispersionCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = DispersionCalculator()
        self.test_data = [1, 2, 3, 4, 5]
        self.test_data_with_outlier = [1, 2, 3, 4, 5, 100]
        self.identical_data = [5, 5, 5, 5, 5]
    
    def test_calculate_range(self):
        """Test range calculation."""
        result = self.calc.calculate_range(self.test_data)
        
        # Check structure
        required_keys = ['range', 'min', 'max', 'midrange']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check values
        self.assertEqual(result['range'], 4.0)
        self.assertEqual(result['min'], 1.0)
        self.assertEqual(result['max'], 5.0)
        self.assertEqual(result['midrange'], 3.0)
        
        # Test with empty data
        with self.assertRaises(ValueError):
            self.calc.calculate_range([])
    
    def test_calculate_iqr(self):
        """Test IQR calculation."""
        # Test with sufficient data
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.calc.calculate_iqr(data)
        
        # Check structure
        required_keys = ['iqr', 'q1', 'q2_median', 'q3', 'lower_fence', 'upper_fence']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that Q3 > Q2 > Q1
        self.assertGreater(result['q3'], result['q2_median'])
        self.assertGreater(result['q2_median'], result['q1'])
        
        # Check IQR calculation
        self.assertEqual(result['iqr'], result['q3'] - result['q1'])
        
        # Test with insufficient data
        with self.assertRaises(ValueError):
            self.calc.calculate_iqr([1, 2, 3])
    
    def test_calculate_variance(self):
        """Test variance calculation."""
        # Test sample variance
        result = self.calc.calculate_variance(self.test_data, sample=True)
        
        # Check structure
        required_keys = ['variance', 'mean', 'n', 'type', 'denominator']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check type
        self.assertEqual(result['type'], 'sample')
        self.assertEqual(result['denominator'], 4)  # n-1 for sample
        
        # Test population variance
        pop_result = self.calc.calculate_variance(self.test_data, sample=False)
        self.assertEqual(pop_result['type'], 'population')
        self.assertEqual(pop_result['denominator'], 5)  # n for population
        
        # Sample variance should be larger than population variance
        self.assertGreater(result['variance'], pop_result['variance'])
        
        # Test with single data point
        with self.assertRaises(ValueError):
            self.calc.calculate_variance([5], sample=True)
        
        # Population variance with single point should be 0
        single_result = self.calc.calculate_variance([5], sample=False)
        self.assertEqual(single_result['variance'], 0.0)
    
    def test_calculate_standard_deviation(self):
        """Test standard deviation calculation."""
        result = self.calc.calculate_standard_deviation(self.test_data, sample=True)
        
        # Check that std_dev is square root of variance
        variance_result = self.calc.calculate_variance(self.test_data, sample=True)
        expected_std = variance_result['variance'] ** 0.5
        self.assertAlmostEqual(result['std_dev'], expected_std, places=10)
        
        # Check that all variance keys are also present
        self.assertIn('variance', result)
        self.assertIn('std_dev', result)
    
    def test_calculate_coefficient_of_variation(self):
        """Test coefficient of variation calculation."""
        result = self.calc.calculate_coefficient_of_variation(self.test_data, sample=True)
        
        # Check structure
        required_keys = ['cv', 'cv_percent', 'interpretation', 'std_dev', 'mean']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check calculation: CV = std_dev / mean
        expected_cv = result['std_dev'] / result['mean']
        self.assertAlmostEqual(result['cv'], expected_cv, places=10)
        
        # Check percentage conversion
        self.assertAlmostEqual(result['cv_percent'], result['cv'] * 100, places=10)
        
        # Test with zero mean
        with self.assertRaises(ValueError):
            self.calc.calculate_coefficient_of_variation([-1, 0, 1])
    
    def test_calculate_mean_absolute_deviation(self):
        """Test mean absolute deviation calculation."""
        # Test MAD from mean
        result_mean = self.calc.calculate_mean_absolute_deviation(self.test_data, 'mean')
        
        # Check structure
        required_keys = ['mad', 'center_type', 'center_value', 'n']
        for key in required_keys:
            self.assertIn(key, result_mean)
        
        self.assertEqual(result_mean['center_type'], 'mean')
        
        # Test MAD from median
        result_median = self.calc.calculate_mean_absolute_deviation(self.test_data, 'median')
        self.assertEqual(result_median['center_type'], 'median')
        
        # Test invalid center
        with self.assertRaises(ValueError):
            self.calc.calculate_mean_absolute_deviation(self.test_data, 'mode')
    
    def test_compare_dispersion_measures(self):
        """Test comparison of dispersion measures."""
        comparison = self.calc.compare_dispersion_measures(self.test_data)
        
        # Should return a DataFrame
        self.assertIsInstance(comparison, pd.DataFrame)
        
        # Should have required columns
        required_columns = ['measure', 'value', 'interpretation', 'robust']
        for col in required_columns:
            self.assertIn(col, comparison.columns)
        
        # Should have multiple measures
        self.assertGreater(len(comparison), 5)
        
        # Check that robust measures are marked correctly
        robust_measures = comparison[comparison['robust'] == True]['measure'].tolist()
        self.assertIn('IQR', robust_measures)
        self.assertIn('MAD (from median)', robust_measures)
    
    def test_analyze_dispersion_impact(self):
        """Test dispersion impact analysis."""
        normal_data = [1, 2, 3, 4, 5]
        skewed_data = [1, 2, 3, 4, 50]
        
        impact = self.calc.analyze_dispersion_impact(normal_data, skewed_data)
        
        # Should be a dictionary
        self.assertIsInstance(impact, dict)
        
        # Should have multiple measures
        self.assertGreater(len(impact), 3)
        
        # Each measure should have required keys
        for measure, stats in impact.items():
            if isinstance(stats, dict):
                required_keys = ['dataset1', 'dataset2', 'difference', 'percent_change', 'higher_dispersion']
                for key in required_keys:
                    self.assertIn(key, stats)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty data
        with self.assertRaises(ValueError):
            self.calc.calculate_range([])
        
        with self.assertRaises(ValueError):
            self.calc.calculate_variance([])
        
        with self.assertRaises(ValueError):
            self.calc.calculate_mean_absolute_deviation([])
        
        # Identical data (zero variance)
        identical_result = self.calc.calculate_variance(self.identical_data, sample=True)
        self.assertEqual(identical_result['variance'], 0.0)
        
        std_result = self.calc.calculate_standard_deviation(self.identical_data, sample=True)
        self.assertEqual(std_result['std_dev'], 0.0)


class TestHackerRankProblems(unittest.TestCase):
    """Test cases for HackerRank-style dispersion problems."""
    
    def test_basic_range_calculation(self):
        """Test basic range calculation."""
        def calculate_range(data):
            if not data:
                return 0
            return max(data) - min(data)
        
        test_data = [10, 5, 8, 12, 3, 15, 7]
        result = calculate_range(test_data)
        expected = 15 - 3  # max - min
        self.assertEqual(result, expected)
    
    def test_iqr_calculation(self):
        """Test IQR calculation implementation."""
        def calculate_iqr_simple(data):
            if len(data) < 4:
                return 0
            
            sorted_data = sorted(data)
            n = len(sorted_data)
            
            # Simple quartile calculation
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            
            return sorted_data[q3_idx] - sorted_data[q1_idx]
        
        test_data = [1, 2, 3, 4, 5, 6, 7, 8]
        result = calculate_iqr_simple(test_data)
        self.assertGreater(result, 0)
    
    def test_variance_difference(self):
        """Test sample vs population variance difference."""
        def variance_difference(data):
            if len(data) <= 1:
                return 0
            
            mean = sum(data) / len(data)
            sum_squared_diff = sum((x - mean) ** 2 for x in data)
            
            population_var = sum_squared_diff / len(data)
            sample_var = sum_squared_diff / (len(data) - 1)
            
            return sample_var - population_var
        
        test_data = [2, 4, 6, 8, 10]
        result = variance_difference(test_data)
        
        # Sample variance should always be larger than population variance
        self.assertGreater(result, 0)
    
    def test_coefficient_of_variation_comparison(self):
        """Test CV comparison between datasets."""
        def calculate_cv(data):
            if not data or len(data) < 2:
                return 0
            
            mean = sum(data) / len(data)
            if mean == 0:
                return float('inf')
            
            variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
            std_dev = variance ** 0.5
            
            return (std_dev / abs(mean)) * 100
        
        # Test with datasets having different scales but similar relative variability
        dataset1 = [100, 110, 120, 130, 140]  # Higher values, same relative spread
        dataset2 = [10, 11, 12, 13, 14]       # Lower values, same relative spread
        
        cv1 = calculate_cv(dataset1)
        cv2 = calculate_cv(dataset2)
        
        # CVs should be approximately equal (same relative variability)
        self.assertAlmostEqual(cv1, cv2, places=1)


def run_interactive_examples():
    """Run interactive examples to demonstrate dispersion concepts."""
    print("=== Interactive Dispersion Examples ===\n")
    
    calc = DispersionCalculator()
    
    # Example 1: Effect of outliers on different measures
    print("Example 1: Effect of Outliers on Dispersion Measures")
    normal_data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    outlier_data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 100]
    
    print(f"Normal data: {normal_data}")
    print(f"Data with outlier: {outlier_data}")
    print()
    
    # Compare measures
    impact = calc.analyze_dispersion_impact(normal_data, outlier_data)
    
    print("Impact of outlier on dispersion measures:")
    for measure, stats in impact.items():
        if isinstance(stats['dataset1'], (int, float)):
            print(f"{measure}:")
            print(f"  Normal: {stats['dataset1']:.2f}")
            print(f"  With outlier: {stats['dataset2']:.2f}")
            print(f"  Change: {stats['difference']:.2f}")
            print()
    
    # Example 2: Comparing datasets with different characteristics
    print("Example 2: Comparing Different Dataset Characteristics")
    
    # Low variability dataset
    low_var = [48, 49, 50, 51, 52]
    # High variability dataset
    high_var = [30, 40, 50, 60, 70]
    # Same mean, different spread
    
    print(f"Low variability: {low_var} (mean: {sum(low_var)/len(low_var)})")
    print(f"High variability: {high_var} (mean: {sum(high_var)/len(high_var)})")
    print()
    
    low_measures = calc.compare_dispersion_measures(low_var)
    high_measures = calc.compare_dispersion_measures(high_var)
    
    print("Dispersion comparison:")
    print("Low variability dataset:")
    print(low_measures[['measure', 'value', 'interpretation']].to_string(index=False))
    print()
    print("High variability dataset:")
    print(high_measures[['measure', 'value', 'interpretation']].to_string(index=False))


if __name__ == "__main__":
    print("Running Dispersion Metrics Tests...\n")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*60 + "\n")
    
    # Run interactive examples
    run_interactive_examples()
