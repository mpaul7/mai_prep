"""
Test suite for error metrics calculations.
Run this to validate your understanding and test your implementations.
"""

import unittest
import numpy as np
import pandas as pd
from error_metrics import ErrorMetricsCalculator
import sys
import os
import math

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestErrorMetricsCalculator(unittest.TestCase):
    """Test cases for the ErrorMetricsCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = ErrorMetricsCalculator()
        
        # Perfect predictions
        self.y_true_perfect = [1, 2, 3, 4, 5]
        self.y_pred_perfect = [1, 2, 3, 4, 5]
        
        # Small errors
        self.y_true_small = [1, 2, 3, 4, 5]
        self.y_pred_small = [1.1, 2.1, 2.9, 4.1, 4.9]
        
        # Large errors
        self.y_true_large = [10, 20, 30, 40, 50]
        self.y_pred_large = [15, 18, 35, 38, 55]
        
        # With zeros (for MAPE testing)
        self.y_true_zeros = [0, 1, 2, 3, 4]
        self.y_pred_zeros = [0.1, 1.1, 1.9, 3.1, 3.9]
    
    def test_mean_squared_error(self):
        """Test MSE calculation."""
        # Perfect predictions should have MSE = 0
        result_perfect = self.calc.mean_squared_error(self.y_true_perfect, self.y_pred_perfect)
        
        # Check structure
        required_keys = ['mse', 'sum_squared_errors', 'n', 'max_squared_error', 'min_squared_error']
        for key in required_keys:
            self.assertIn(key, result_perfect)
        
        # Perfect predictions should have MSE = 0
        self.assertEqual(result_perfect['mse'], 0.0)
        self.assertEqual(result_perfect['sum_squared_errors'], 0.0)
        
        # Small errors should have small MSE
        result_small = self.calc.mean_squared_error(self.y_true_small, self.y_pred_small)
        self.assertGreater(result_small['mse'], 0.0)
        self.assertLess(result_small['mse'], 1.0)
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.calc.mean_squared_error([1, 2, 3], [1, 2])
        
        # Test with empty arrays
        with self.assertRaises(ValueError):
            self.calc.mean_squared_error([], [])
    
    def test_root_mean_squared_error(self):
        """Test RMSE calculation."""
        result = self.calc.root_mean_squared_error(self.y_true_small, self.y_pred_small)
        
        # Check structure
        required_keys = ['rmse', 'mse', 'nrmse_range', 'nrmse_mean']
        for key in required_keys:
            self.assertIn(key, result)
        
        # RMSE should be square root of MSE
        expected_rmse = math.sqrt(result['mse'])
        self.assertAlmostEqual(result['rmse'], expected_rmse, places=10)
        
        # Perfect predictions should have RMSE = 0
        result_perfect = self.calc.root_mean_squared_error(self.y_true_perfect, self.y_pred_perfect)
        self.assertEqual(result_perfect['rmse'], 0.0)
    
    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        result = self.calc.mean_absolute_error(self.y_true_small, self.y_pred_small)
        
        # Check structure
        required_keys = ['mae', 'sum_absolute_errors', 'n', 'max_absolute_error', 'median_absolute_error']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Perfect predictions should have MAE = 0
        result_perfect = self.calc.mean_absolute_error(self.y_true_perfect, self.y_pred_perfect)
        self.assertEqual(result_perfect['mae'], 0.0)
        
        # MAE should be non-negative
        self.assertGreaterEqual(result['mae'], 0.0)
        
        # Manual calculation check
        manual_mae = sum(abs(t - p) for t, p in zip(self.y_true_small, self.y_pred_small)) / len(self.y_true_small)
        self.assertAlmostEqual(result['mae'], manual_mae, places=10)
    
    def test_mean_absolute_percentage_error(self):
        """Test MAPE calculation."""
        result = self.calc.mean_absolute_percentage_error(self.y_true_small, self.y_pred_small)
        
        # Check structure
        required_keys = ['mape', 'n', 'zero_count', 'valid_count', 'interpretation']
        for key in required_keys:
            self.assertIn(key, result)
        
        # MAPE should be non-negative
        self.assertGreaterEqual(result['mape'], 0.0)
        
        # Test with zeros in y_true
        result_zeros = self.calc.mean_absolute_percentage_error(self.y_true_zeros, self.y_pred_zeros)
        self.assertGreater(result_zeros['zero_count'], 0)
        self.assertLess(result_zeros['valid_count'], len(self.y_true_zeros))
        
        # Test with all zeros
        with self.assertRaises(ValueError):
            self.calc.mean_absolute_percentage_error([0, 0, 0], [1, 1, 1])
    
    def test_symmetric_mean_absolute_percentage_error(self):
        """Test SMAPE calculation."""
        result = self.calc.symmetric_mean_absolute_percentage_error(self.y_true_small, self.y_pred_small)
        
        # Check structure
        required_keys = ['smape', 'n', 'zero_denominator_count', 'valid_count', 'interpretation']
        for key in required_keys:
            self.assertIn(key, result)
        
        # SMAPE should be between 0 and 100
        self.assertGreaterEqual(result['smape'], 0.0)
        self.assertLessEqual(result['smape'], 100.0)
        
        # Perfect predictions should have SMAPE = 0
        result_perfect = self.calc.symmetric_mean_absolute_percentage_error(self.y_true_perfect, self.y_pred_perfect)
        self.assertEqual(result_perfect['smape'], 0.0)
    
    def test_mean_absolute_scaled_error(self):
        """Test MASE calculation."""
        # Need more data points for MASE
        y_true_long = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_pred_long = [1.1, 2.1, 2.9, 4.1, 4.9, 6.1, 6.9, 8.1, 8.9, 10.1]
        
        result = self.calc.mean_absolute_scaled_error(y_true_long, y_pred_long)
        
        # Check structure
        required_keys = ['mase', 'mae_model', 'mae_naive', 'n', 'interpretation']
        for key in required_keys:
            self.assertIn(key, result)
        
        # MASE should be positive
        self.assertGreater(result['mase'], 0.0)
        
        # Test with training data
        y_train = [0, 1, 2, 3, 4, 5]
        result_with_train = self.calc.mean_absolute_scaled_error(y_true_long, y_pred_long, y_train)
        self.assertIn('mase', result_with_train)
        
        # Test with insufficient data
        with self.assertRaises(ValueError):
            self.calc.mean_absolute_scaled_error([1], [1])
    
    def test_mean_squared_logarithmic_error(self):
        """Test MSLE calculation."""
        # Use positive values for MSLE
        y_true_pos = [1, 2, 3, 4, 5]
        y_pred_pos = [1.1, 2.1, 2.9, 4.1, 4.9]
        
        result = self.calc.mean_squared_logarithmic_error(y_true_pos, y_pred_pos)
        
        # Check structure
        required_keys = ['msle', 'rmsle', 'sum_squared_log_errors', 'n']
        for key in required_keys:
            self.assertIn(key, result)
        
        # MSLE should be non-negative
        self.assertGreaterEqual(result['msle'], 0.0)
        
        # RMSLE should be square root of MSLE
        self.assertAlmostEqual(result['rmsle'], math.sqrt(result['msle']), places=10)
        
        # Test with negative values (should raise error)
        with self.assertRaises(ValueError):
            self.calc.mean_squared_logarithmic_error([-1, 2, 3], [1, 2, 3])
    
    def test_r_squared(self):
        """Test R-squared calculation."""
        result = self.calc.r_squared(self.y_true_small, self.y_pred_small)
        
        # Check structure
        required_keys = ['r_squared', 'ss_res', 'ss_tot', 'y_mean', 'n', 'interpretation']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Perfect predictions should have R² = 1
        result_perfect = self.calc.r_squared(self.y_true_perfect, self.y_pred_perfect)
        self.assertAlmostEqual(result_perfect['r_squared'], 1.0, places=10)
        
        # R² should be <= 1 for good predictions
        self.assertLessEqual(result['r_squared'], 1.0)
        
        # Test with constant y_true (SS_tot = 0)
        y_true_constant = [5, 5, 5, 5, 5]
        y_pred_constant = [5, 5, 5, 5, 5]
        result_constant = self.calc.r_squared(y_true_constant, y_pred_constant)
        self.assertEqual(result_constant['r_squared'], 1.0)
    
    def test_adjusted_r_squared(self):
        """Test Adjusted R-squared calculation."""
        result = self.calc.adjusted_r_squared(self.y_true_small, self.y_pred_small, n_features=1)
        
        # Check structure
        required_keys = ['adj_r_squared', 'r_squared', 'n_features', 'degrees_of_freedom']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Adjusted R² should be less than or equal to R² for positive R²
        if result['r_squared'] > 0:
            self.assertLessEqual(result['adj_r_squared'], result['r_squared'])
        
        # Test with insufficient degrees of freedom
        result_insufficient = self.calc.adjusted_r_squared([1, 2], [1, 2], n_features=2)
        self.assertEqual(result_insufficient['adj_r_squared'], float('-inf'))
    
    def test_mean_directional_accuracy(self):
        """Test MDA calculation."""
        # Create data with known directional changes
        y_true_trend = [1, 2, 3, 4, 5]
        y_pred_trend = [1.1, 2.1, 3.1, 4.1, 5.1]  # Same trend
        
        result = self.calc.mean_directional_accuracy(y_true_trend, y_pred_trend)
        
        # Check structure
        required_keys = ['mda', 'correct_predictions', 'total_predictions', 'n', 'interpretation']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Should have 100% directional accuracy for same trend
        self.assertEqual(result['mda'], 100.0)
        
        # Test with opposite trend
        y_pred_opposite = [5.1, 4.1, 3.1, 2.1, 1.1]
        result_opposite = self.calc.mean_directional_accuracy(y_true_trend, y_pred_opposite)
        self.assertEqual(result_opposite['mda'], 0.0)
        
        # Test with insufficient data
        with self.assertRaises(ValueError):
            self.calc.mean_directional_accuracy([1], [1])
    
    def test_theil_u_statistic(self):
        """Test Theil's U statistic calculation."""
        result = self.calc.theil_u_statistic(self.y_true_small, self.y_pred_small)
        
        # Check structure
        required_keys = ['theil_u', 'rmse', 'mean_true_squared', 'mean_pred_squared', 'n', 'interpretation']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Theil's U should be non-negative
        self.assertGreaterEqual(result['theil_u'], 0.0)
        
        # Perfect predictions should have Theil's U = 0
        result_perfect = self.calc.theil_u_statistic(self.y_true_perfect, self.y_pred_perfect)
        self.assertEqual(result_perfect['theil_u'], 0.0)
    
    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        result = self.calc.calculate_all_metrics(self.y_true_small, self.y_pred_small, n_features=1)
        
        # Should return a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Should have required columns
        required_columns = ['metric', 'value', 'interpretation']
        for col in required_columns:
            self.assertIn(col, result.columns)
        
        # Should have multiple metrics
        self.assertGreater(len(result), 5)
        
        # Check that some key metrics are present
        metrics = result['metric'].tolist()
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('R²', metrics)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty arrays
        with self.assertRaises(ValueError):
            self.calc.mean_squared_error([], [])
        
        with self.assertRaises(ValueError):
            self.calc.mean_absolute_error([], [])
        
        # Mismatched lengths
        with self.assertRaises(ValueError):
            self.calc.mean_squared_error([1, 2, 3], [1, 2])
        
        with self.assertRaises(ValueError):
            self.calc.mean_absolute_error([1, 2, 3], [1, 2])
        
        # Single data point
        single_true = [5]
        single_pred = [5.1]
        
        # Should work for most metrics
        mse_result = self.calc.mean_squared_error(single_true, single_pred)
        self.assertGreater(mse_result['mse'], 0.0)
        
        # Should fail for metrics requiring multiple points
        with self.assertRaises(ValueError):
            self.calc.mean_directional_accuracy(single_true, single_pred)


class TestHackerRankProblems(unittest.TestCase):
    """Test cases for HackerRank-style error metrics problems."""
    
    def test_mse_calculation(self):
        """Test MSE implementation."""
        def calculate_mse(y_true, y_pred):
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
            return sum(squared_errors) / len(squared_errors)
        
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.2, 2.8, 4.1, 4.9]
        result = calculate_mse(y_true, y_pred)
        
        # Manual calculation
        expected = ((1-1.1)**2 + (2-2.2)**2 + (3-2.8)**2 + (4-4.1)**2 + (5-4.9)**2) / 5
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_rmse_calculation(self):
        """Test RMSE implementation."""
        def calculate_rmse(y_true, y_pred):
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            mse = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)
            return mse ** 0.5
        
        y_true = [10, 20, 30, 40, 50]
        y_pred = [12, 18, 32, 38, 52]
        result = calculate_rmse(y_true, y_pred)
        
        # Should be positive
        self.assertGreater(result, 0.0)
        
        # Should be square root of MSE
        mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
        expected_rmse = mse ** 0.5
        self.assertAlmostEqual(result, expected_rmse, places=10)
    
    def test_mae_calculation(self):
        """Test MAE implementation."""
        def calculate_mae(y_true, y_pred):
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            absolute_errors = [abs(true - pred) for true, pred in zip(y_true, y_pred)]
            return sum(absolute_errors) / len(absolute_errors)
        
        y_true = [1, 3, 5, 7, 9]
        y_pred = [2, 3, 4, 8, 10]
        result = calculate_mae(y_true, y_pred)
        
        # Manual calculation
        expected = (abs(1-2) + abs(3-3) + abs(5-4) + abs(7-8) + abs(9-10)) / 5
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_mape_calculation(self):
        """Test MAPE implementation."""
        def calculate_mape(y_true, y_pred):
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            
            percentage_errors = []
            for true, pred in zip(y_true, y_pred):
                if true != 0:
                    percentage_errors.append(abs((true - pred) / true) * 100)
            
            if not percentage_errors:
                return float('inf')
            
            return sum(percentage_errors) / len(percentage_errors)
        
        y_true = [100, 200, 300, 400, 500]
        y_pred = [110, 190, 320, 380, 520]
        result = calculate_mape(y_true, y_pred)
        
        # Should be positive
        self.assertGreater(result, 0.0)
        
        # Manual calculation
        expected = (10 + 5 + 20/3 + 5 + 4) / 5  # Percentage errors
        self.assertAlmostEqual(result, expected, places=1)
    
    def test_r_squared_calculation(self):
        """Test R-squared implementation."""
        def calculate_r_squared(y_true, y_pred):
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            
            y_mean = sum(y_true) / len(y_true)
            ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
            ss_tot = sum((true - y_mean) ** 2 for true in y_true)
            
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            
            return 1 - (ss_res / ss_tot)
        
        # Perfect predictions
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        result = calculate_r_squared(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0, places=10)
        
        # Imperfect predictions
        y_pred_imperfect = [1.1, 2.1, 2.9, 4.1, 4.9]
        result_imperfect = calculate_r_squared(y_true, y_pred_imperfect)
        self.assertLess(result_imperfect, 1.0)
        self.assertGreater(result_imperfect, 0.8)  # Should still be high


def run_interactive_examples():
    """Run interactive examples to demonstrate error metrics concepts."""
    print("=== Interactive Error Metrics Examples ===\n")
    
    calc = ErrorMetricsCalculator()
    
    # Example 1: Different prediction qualities
    print("Example 1: Different Prediction Qualities")
    
    y_true = [10, 20, 30, 40, 50]
    
    # Perfect predictions
    y_pred_perfect = [10, 20, 30, 40, 50]
    # Good predictions
    y_pred_good = [10.5, 19.8, 30.2, 39.7, 50.3]
    # Poor predictions
    y_pred_poor = [12, 18, 32, 38, 52]
    
    predictions = [
        ("Perfect", y_pred_perfect),
        ("Good", y_pred_good),
        ("Poor", y_pred_poor)
    ]
    
    for name, y_pred in predictions:
        mse = calc.mean_squared_error(y_true, y_pred)
        mae = calc.mean_absolute_error(y_true, y_pred)
        r2 = calc.r_squared(y_true, y_pred)
        
        print(f"{name} Predictions:")
        print(f"  Predicted: {y_pred}")
        print(f"  MSE: {mse['mse']:.3f}")
        print(f"  MAE: {mae['mae']:.3f}")
        print(f"  R²: {r2['r_squared']:.3f} ({r2['interpretation']})")
        print()
    
    # Example 2: Metric sensitivity comparison
    print("Example 2: Metric Sensitivity to Outliers")
    
    y_true_base = [1, 2, 3, 4, 5]
    y_pred_base = [1.1, 2.1, 2.9, 4.1, 5.1]  # Small errors
    y_pred_outlier = [1.1, 2.1, 2.9, 4.1, 10.0]  # One large error
    
    scenarios = [
        ("Small Errors", y_pred_base),
        ("With Outlier", y_pred_outlier)
    ]
    
    for name, y_pred in scenarios:
        mse = calc.mean_squared_error(y_true_base, y_pred)
        mae = calc.mean_absolute_error(y_true_base, y_pred)
        
        print(f"{name}:")
        print(f"  Predicted: {y_pred}")
        print(f"  MSE: {mse['mse']:.3f}")
        print(f"  MAE: {mae['mae']:.3f}")
        print(f"  MSE/MAE Ratio: {mse['mse']/mae['mae']:.2f}")
        print()
    
    # Example 3: Percentage-based metrics
    print("Example 3: Percentage-based Metrics")
    
    # Different scales
    small_scale = ([1, 2, 3, 4, 5], [1.1, 2.1, 2.9, 4.1, 4.9])
    large_scale = ([100, 200, 300, 400, 500], [110, 210, 290, 410, 490])
    
    scales = [("Small Scale", small_scale), ("Large Scale", large_scale)]
    
    for name, (y_true, y_pred) in scales:
        mae = calc.mean_absolute_error(y_true, y_pred)
        mape = calc.mean_absolute_percentage_error(y_true, y_pred)
        
        print(f"{name}:")
        print(f"  True: {y_true}")
        print(f"  Pred: {y_pred}")
        print(f"  MAE: {mae['mae']:.2f}")
        print(f"  MAPE: {mape['mape']:.2f}% ({mape['interpretation']})")
        print()


if __name__ == "__main__":
    print("Running Error Metrics Tests...\n")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*60 + "\n")
    
    # Run interactive examples
    run_interactive_examples()
