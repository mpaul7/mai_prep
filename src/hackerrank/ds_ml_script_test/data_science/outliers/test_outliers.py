"""
Test suite for outlier detection methods.
Run this to validate your understanding and test your implementations.
"""

import unittest
import numpy as np
import pandas as pd
from outliers import OutlierDetector
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestOutlierDetector(unittest.TestCase):
    """Test cases for the OutlierDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = OutlierDetector()
        # Normal data with clear outliers
        self.test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        # Data without outliers
        self.normal_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Data with extreme outliers
        self.extreme_data = [10, 12, 14, 16, 18, 20, 1000, -500]
    
    def test_z_score_outliers(self):
        """Test Z-score outlier detection."""
        result = self.detector.z_score_outliers(self.test_data, threshold=2.0)
        
        # Check structure
        required_keys = ['outliers', 'outlier_indices', 'z_scores', 'threshold', 'mean', 'std']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Should detect the outlier (100)
        self.assertGreater(len(result['outliers']), 0)
        self.assertIn(100, result['outliers'])
        
        # Check that z_scores length matches data length
        self.assertEqual(len(result['z_scores']), len(self.test_data))
    
    def test_modified_z_score_outliers(self):
        """Test Modified Z-score outlier detection."""
        result = self.detector.modified_z_score_outliers(self.test_data)
        
        # Check structure
        required_keys = ['outliers', 'outlier_indices', 'modified_z_scores', 'threshold', 'median', 'mad']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Should detect outliers
        self.assertGreater(len(result['outliers']), 0)
    
    def test_iqr_outliers(self):
        """Test IQR outlier detection."""
        result = self.detector.iqr_outliers(self.test_data)
        
        # Check structure
        required_keys = ['outliers', 'outlier_indices', 'lower_bound', 'upper_bound', 'q1', 'q3', 'iqr', 'k']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Should detect the outlier (100)
        self.assertGreater(len(result['outliers']), 0)
        self.assertIn(100, result['outliers'])
        
        # Check that bounds are calculated correctly
        self.assertLess(result['lower_bound'], result['upper_bound'])
        self.assertLess(result['q1'], result['q3'])
    
    def test_percentile_outliers(self):
        """Test percentile-based outlier detection."""
        result = self.detector.percentile_outliers(self.test_data, 10, 90)
        
        # Check structure
        required_keys = ['outliers', 'outlier_indices', 'lower_threshold', 'upper_threshold']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Should detect outliers
        self.assertGreater(len(result['outliers']), 0)
    
    def test_isolation_forest_outliers(self):
        """Test Isolation Forest outlier detection."""
        # Create larger dataset for Isolation Forest
        np.random.seed(42)
        large_data = np.random.normal(0, 1, 50).tolist() + [10, -10]
        
        result = self.detector.isolation_forest_outliers(large_data)
        
        # Check structure
        required_keys = ['outliers', 'outlier_indices', 'anomaly_scores', 'outlier_labels']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that scores and labels have correct length
        self.assertEqual(len(result['anomaly_scores']), len(large_data))
        self.assertEqual(len(result['outlier_labels']), len(large_data))
    
    def test_compare_methods(self):
        """Test comparison of different methods."""
        comparison = self.detector.compare_methods(self.test_data)
        
        # Should return a DataFrame
        self.assertIsInstance(comparison, pd.DataFrame)
        
        # Should have method column
        self.assertIn('method', comparison.columns)
        self.assertIn('outliers_count', comparison.columns)
        
        # Should have multiple methods
        self.assertGreater(len(comparison), 2)
    
    def test_analyze_outlier_impact(self):
        """Test outlier impact analysis."""
        # First detect outliers
        iqr_result = self.detector.iqr_outliers(self.test_data)
        
        # Then analyze impact
        impact = self.detector.analyze_outlier_impact(self.test_data, iqr_result['outlier_indices'])
        
        # Check structure
        required_keys = ['original_mean', 'clean_mean', 'mean_change', 'original_std', 'clean_std']
        for key in required_keys:
            self.assertIn(key, impact)
        
        # Mean should change when outliers are removed
        self.assertNotEqual(impact['original_mean'], impact['clean_mean'])
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty data
        with self.assertRaises(ValueError):
            self.detector.z_score_outliers([])
        
        # Single data point
        with self.assertRaises(ValueError):
            self.detector.z_score_outliers([1])
        
        # Insufficient data for IQR
        with self.assertRaises(ValueError):
            self.detector.iqr_outliers([1, 2])
        
        # All identical values
        identical_data = [5, 5, 5, 5, 5]
        result = self.detector.z_score_outliers(identical_data)
        self.assertEqual(len(result['outliers']), 0)  # No outliers when std = 0


class TestHackerRankProblems(unittest.TestCase):
    """Test cases for HackerRank-style outlier problems."""
    
    def test_z_score_outlier_count(self):
        """Test basic Z-score outlier counting."""
        def count_z_score_outliers(data, threshold=2.0):
            if len(data) < 2:
                return 0
            
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
            std = variance ** 0.5
            
            if std == 0:
                return 0
            
            outliers = 0
            for x in data:
                z_score = abs((x - mean) / std)
                if z_score > threshold:
                    outliers += 1
            
            return outliers
        
        test_data = [1, 2, 3, 4, 5, 100, 2, 3, 4, 5]
        result = count_z_score_outliers(test_data, 2.0)
        self.assertGreater(result, 0)  # Should find at least one outlier
    
    def test_iqr_outlier_detection(self):
        """Test IQR outlier detection implementation."""
        def find_iqr_outliers(data, k=1.5):
            if len(data) < 4:
                return []
            
            sorted_data = sorted(data)
            n = len(sorted_data)
            
            q1_index = n // 4
            q3_index = 3 * n // 4
            
            q1 = sorted_data[q1_index]
            q3 = sorted_data[q3_index]
            iqr = q3 - q1
            
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            
            outliers = [x for x in data if x < lower_bound or x > upper_bound]
            return outliers
        
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        result = find_iqr_outliers(test_data)
        self.assertIn(100, result)  # Should detect 100 as outlier
    
    def test_outlier_percentage_calculation(self):
        """Test outlier percentage calculation."""
        def calculate_outlier_percentage(data, k=1.5):
            if len(data) < 4:
                return 0.0
            
            sorted_data = sorted(data)
            n = len(sorted_data)
            
            q1 = sorted_data[n//4]
            q3 = sorted_data[3*n//4]
            iqr = q3 - q1
            
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            
            outlier_count = sum(1 for x in data if x < lower_bound or x > upper_bound)
            percentage = (outlier_count / len(data)) * 100
            
            return round(percentage, 2)
        
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300]
        result = calculate_outlier_percentage(test_data)
        self.assertGreater(result, 0)  # Should find some outliers
        self.assertLessEqual(result, 100)  # Percentage should be <= 100


def run_interactive_examples():
    """Run interactive examples to demonstrate outlier concepts."""
    print("=== Interactive Outlier Detection Examples ===\n")
    
    # Example 1: Different types of outliers
    print("Example 1: Different Types of Outliers")
    detector = OutlierDetector()
    
    # Mild outliers
    mild_outliers = [10, 12, 14, 16, 18, 20, 22, 24, 26, 35]
    print(f"Mild outliers data: {mild_outliers}")
    
    iqr_mild = detector.iqr_outliers(mild_outliers)
    print(f"IQR method detects: {iqr_mild['outliers']}")
    
    z_mild = detector.z_score_outliers(mild_outliers, threshold=2.0)
    print(f"Z-score method detects: {z_mild['outliers']}")
    print()
    
    # Example 2: Impact of outliers on different statistics
    print("Example 2: Impact on Statistics")
    data_with_outliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    data_without_outliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print(f"With outliers: {data_with_outliers}")
    print(f"Mean: {np.mean(data_with_outliers):.2f}, Median: {np.median(data_with_outliers):.2f}")
    print(f"Std: {np.std(data_with_outliers, ddof=1):.2f}")
    
    print(f"Without outliers: {data_without_outliers}")
    print(f"Mean: {np.mean(data_without_outliers):.2f}, Median: {np.median(data_without_outliers):.2f}")
    print(f"Std: {np.std(data_without_outliers, ddof=1):.2f}")
    print()
    
    # Example 3: Method comparison
    print("Example 3: Method Comparison")
    test_data = [5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60]
    print(f"Test data: {test_data}")
    
    comparison = detector.compare_methods(test_data)
    print("Method comparison:")
    print(comparison[['method', 'outliers_count', 'outliers']].to_string(index=False))


if __name__ == "__main__":
    print("Running Outlier Detection Tests...\n")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*60 + "\n")
    
    # Run interactive examples
    run_interactive_examples()

