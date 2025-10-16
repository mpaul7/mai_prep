"""
Test suite for bivariate analysis methods.
Run this to validate your understanding and test your implementations.
"""

import unittest
import numpy as np
import pandas as pd
from bivariate import BivariateAnalyzer
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestBivariateAnalyzer(unittest.TestCase):
    """Test cases for the BivariateAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BivariateAnalyzer()
        
        # Perfect positive correlation
        self.x_perfect = [1, 2, 3, 4, 5]
        self.y_perfect = [2, 4, 6, 8, 10]
        
        # Perfect negative correlation
        self.x_negative = [1, 2, 3, 4, 5]
        self.y_negative = [10, 8, 6, 4, 2]
        
        # No correlation
        self.x_no_corr = [1, 2, 3, 4, 5]
        self.y_no_corr = [3, 1, 4, 2, 5]
        
        # Identical values (no variance)
        self.x_identical = [5, 5, 5, 5, 5]
        self.y_identical = [3, 3, 3, 3, 3]
    
    def test_calculate_covariance(self):
        """Test covariance calculation."""
        # Test sample covariance
        result = self.analyzer.calculate_covariance(self.x_perfect, self.y_perfect, sample=True)
        
        # Check structure
        required_keys = ['covariance', 'x_mean', 'y_mean', 'n', 'type', 'denominator']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check type
        self.assertEqual(result['type'], 'sample')
        self.assertEqual(result['denominator'], 4)  # n-1 for sample
        
        # Test population covariance
        pop_result = self.analyzer.calculate_covariance(self.x_perfect, self.y_perfect, sample=False)
        self.assertEqual(pop_result['type'], 'population')
        self.assertEqual(pop_result['denominator'], 5)  # n for population
        
        # Sample covariance should be larger than population covariance
        self.assertGreater(result['covariance'], pop_result['covariance'])
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.analyzer.calculate_covariance([1, 2, 3], [1, 2])
        
        # Test with insufficient data
        with self.assertRaises(ValueError):
            self.analyzer.calculate_covariance([1], [1], sample=True)
    
    def test_calculate_pearson_correlation(self):
        """Test Pearson correlation calculation."""
        # Test perfect positive correlation
        result = self.analyzer.calculate_pearson_correlation(self.x_perfect, self.y_perfect)
        
        # Check structure
        required_keys = ['correlation', 'interpretation', 'strength', 'direction', 'n']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Should be perfect correlation
        self.assertAlmostEqual(result['correlation'], 1.0, places=10)
        self.assertEqual(result['direction'], 'positive')
        
        # Test perfect negative correlation
        neg_result = self.analyzer.calculate_pearson_correlation(self.x_negative, self.y_negative)
        self.assertAlmostEqual(neg_result['correlation'], -1.0, places=10)
        self.assertEqual(neg_result['direction'], 'negative')
        
        # Test no correlation (should be close to 0)
        no_corr_result = self.analyzer.calculate_pearson_correlation(self.x_no_corr, self.y_no_corr)
        self.assertLess(abs(no_corr_result['correlation']), 0.5)  # Should be weak correlation
        
        # Test identical values (no variance)
        identical_result = self.analyzer.calculate_pearson_correlation(self.x_identical, self.y_identical)
        self.assertEqual(identical_result['correlation'], 0.0)
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.analyzer.calculate_pearson_correlation([1, 2, 3], [1, 2])
    
    def test_calculate_spearman_correlation(self):
        """Test Spearman correlation calculation."""
        # Test with monotonic relationship
        x_mono = [1, 2, 3, 4, 5]
        y_mono = [1, 4, 9, 16, 25]  # x^2 relationship
        
        result = self.analyzer.calculate_spearman_correlation(x_mono, y_mono)
        
        # Check structure
        required_keys = ['correlation', 'interpretation', 'strength', 'direction', 'type', 'n']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Should be perfect Spearman correlation (monotonic)
        self.assertAlmostEqual(result['correlation'], 1.0, places=10)
        self.assertEqual(result['type'], 'spearman')
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.analyzer.calculate_spearman_correlation([1, 2, 3], [1, 2])
    
    def test_simple_linear_regression(self):
        """Test simple linear regression."""
        result = self.analyzer.simple_linear_regression(self.x_perfect, self.y_perfect)
        
        # Check structure
        required_keys = ['slope', 'intercept', 'r_squared', 'predictions', 'residuals', 'equation']
        for key in required_keys:
            self.assertIn(key, result)
        
        # For y = 2x, slope should be 2, intercept should be 0
        self.assertAlmostEqual(result['slope'], 2.0, places=10)
        self.assertAlmostEqual(result['intercept'], 0.0, places=10)
        
        # R-squared should be 1 for perfect fit
        self.assertAlmostEqual(result['r_squared'], 1.0, places=10)
        
        # Check predictions
        expected_predictions = [2, 4, 6, 8, 10]
        for pred, expected in zip(result['predictions'], expected_predictions):
            self.assertAlmostEqual(pred, expected, places=10)
        
        # Test with no variance in x
        with self.assertRaises(ValueError):
            self.analyzer.simple_linear_regression(self.x_identical, self.y_perfect)
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.analyzer.simple_linear_regression([1, 2, 3], [1, 2])
    
    def test_calculate_goodness_of_fit(self):
        """Test goodness of fit calculations."""
        # Perfect predictions
        y_true = [1, 2, 3, 4, 5]
        y_pred_perfect = [1, 2, 3, 4, 5]
        
        result = self.analyzer.calculate_goodness_of_fit(y_true, y_pred_perfect)
        
        # Check structure
        required_keys = ['r_squared', 'adj_r_squared', 'mse', 'rmse', 'mae', 'mape', 'interpretation']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Perfect predictions should have R² = 1, MSE = 0, etc.
        self.assertAlmostEqual(result['r_squared'], 1.0, places=10)
        self.assertAlmostEqual(result['mse'], 0.0, places=10)
        self.assertAlmostEqual(result['rmse'], 0.0, places=10)
        self.assertAlmostEqual(result['mae'], 0.0, places=10)
        
        # Test with imperfect predictions
        y_pred_imperfect = [1.1, 2.1, 2.9, 4.1, 4.9]
        imperfect_result = self.analyzer.calculate_goodness_of_fit(y_true, y_pred_imperfect)
        
        # Should have good but not perfect fit
        self.assertLess(imperfect_result['r_squared'], 1.0)
        self.assertGreater(imperfect_result['r_squared'], 0.8)
        self.assertGreater(imperfect_result['mse'], 0.0)
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.analyzer.calculate_goodness_of_fit([1, 2, 3], [1, 2])
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        data = {
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10],  # Perfect positive correlation with x
            'z': [5, 4, 3, 2, 1]   # Perfect negative correlation with x
        }
        
        corr_matrix = self.analyzer.correlation_matrix(data)
        
        # Should be a DataFrame
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        
        # Should be square matrix
        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
        
        # Diagonal should be 1.0
        for i in range(len(corr_matrix)):
            self.assertAlmostEqual(corr_matrix.iloc[i, i], 1.0, places=10)
        
        # Should be symmetric
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                self.assertAlmostEqual(corr_matrix.iloc[i, j], corr_matrix.iloc[j, i], places=10)
        
        # Check specific correlations
        self.assertAlmostEqual(corr_matrix.loc['x', 'y'], 1.0, places=10)  # Perfect positive
        self.assertAlmostEqual(corr_matrix.loc['x', 'z'], -1.0, places=10)  # Perfect negative
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty data
        with self.assertRaises(ValueError):
            self.analyzer.calculate_covariance([], [])
        
        with self.assertRaises(ValueError):
            self.analyzer.calculate_pearson_correlation([], [])
        
        with self.assertRaises(ValueError):
            self.analyzer.simple_linear_regression([], [])
        
        with self.assertRaises(ValueError):
            self.analyzer.calculate_goodness_of_fit([], [])
        
        # Single data point
        with self.assertRaises(ValueError):
            self.analyzer.calculate_covariance([1], [1], sample=True)
        
        with self.assertRaises(ValueError):
            self.analyzer.calculate_pearson_correlation([1], [1])
        
        with self.assertRaises(ValueError):
            self.analyzer.simple_linear_regression([1], [1])


class TestHackerRankProblems(unittest.TestCase):
    """Test cases for HackerRank-style bivariate problems."""
    
    def test_pearson_correlation_calculation(self):
        """Test Pearson correlation implementation."""
        def calculate_pearson(x, y):
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            sum_y2 = sum(y[i] ** 2 for i in range(n))
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        
        # Test perfect correlation
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = calculate_pearson(x, y)
        self.assertAlmostEqual(result, 1.0, places=10)
        
        # Test negative correlation
        y_neg = [10, 8, 6, 4, 2]
        result_neg = calculate_pearson(x, y_neg)
        self.assertAlmostEqual(result_neg, -1.0, places=10)
    
    def test_covariance_calculation(self):
        """Test covariance implementation."""
        def calculate_sample_covariance(x, y):
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            n = len(x)
            x_mean = sum(x) / n
            y_mean = sum(y) / n
            
            covariance = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / (n - 1)
            return covariance
        
        x = [1, 3, 5, 7, 9]
        y = [2, 6, 10, 14, 18]
        result = calculate_sample_covariance(x, y)
        
        # Should be positive covariance
        self.assertGreater(result, 0)
        
        # Manual calculation check
        expected = 20.0  # Based on the linear relationship
        self.assertAlmostEqual(result, expected, places=1)
    
    def test_linear_regression_calculation(self):
        """Test linear regression implementation."""
        def calculate_regression(x, y):
            if len(x) != len(y) or len(x) < 2:
                return 0.0, 0.0
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Calculate intercept
            intercept = (sum_y - slope * sum_x) / n
            
            return slope, intercept
        
        # Test with y = 2x + 1
        x = [1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11]
        slope, intercept = calculate_regression(x, y)
        
        self.assertAlmostEqual(slope, 2.0, places=10)
        self.assertAlmostEqual(intercept, 1.0, places=10)
    
    def test_r_squared_calculation(self):
        """Test R-squared implementation."""
        def calculate_r_squared(y_actual, y_predicted):
            if len(y_actual) != len(y_predicted) or len(y_actual) == 0:
                return 0.0
            
            y_mean = sum(y_actual) / len(y_actual)
            
            ss_res = sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual)))
            ss_tot = sum((y_actual[i] - y_mean) ** 2 for i in range(len(y_actual)))
            
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            
            return 1 - (ss_res / ss_tot)
        
        # Perfect predictions
        y_actual = [1, 2, 3, 4, 5]
        y_predicted = [1, 2, 3, 4, 5]
        result = calculate_r_squared(y_actual, y_predicted)
        self.assertAlmostEqual(result, 1.0, places=10)
        
        # Imperfect predictions
        y_predicted_imperfect = [1.1, 2.2, 2.8, 4.1, 4.9]
        result_imperfect = calculate_r_squared(y_actual, y_predicted_imperfect)
        self.assertLess(result_imperfect, 1.0)
        self.assertGreater(result_imperfect, 0.8)


def run_interactive_examples():
    """Run interactive examples to demonstrate bivariate concepts."""
    print("=== Interactive Bivariate Analysis Examples ===\n")
    
    analyzer = BivariateAnalyzer()
    
    # Example 1: Different types of relationships
    print("Example 1: Different Types of Relationships")
    
    # Linear relationship
    x_linear = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_linear = [2*x + 1 + np.random.normal(0, 0.5) for x in x_linear]
    
    # Quadratic relationship
    x_quad = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_quad = [x**2 for x in x_quad]
    
    # No relationship
    np.random.seed(42)
    x_random = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_random = np.random.normal(50, 10, 10).tolist()
    
    relationships = [
        ("Linear", x_linear, y_linear),
        ("Quadratic", x_quad, y_quad),
        ("Random", x_random, y_random)
    ]
    
    for name, x, y in relationships:
        pearson = analyzer.calculate_pearson_correlation(x, y)
        spearman = analyzer.calculate_spearman_correlation(x, y)
        
        print(f"{name} Relationship:")
        print(f"  Pearson r: {pearson['correlation']:.3f} ({pearson['interpretation']})")
        print(f"  Spearman ρ: {spearman['correlation']:.3f} ({spearman['interpretation']})")
        print()
    
    # Example 2: Regression analysis
    print("Example 2: Regression Analysis")
    x_reg = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_reg = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]  # y = 2x + 1
    
    regression = analyzer.simple_linear_regression(x_reg, y_reg)
    gof = analyzer.calculate_goodness_of_fit(y_reg, regression['predictions'])
    
    print(f"Data: x = {x_reg}")
    print(f"      y = {y_reg}")
    print(f"Regression equation: {regression['equation']}")
    print(f"R-squared: {gof['r_squared']:.4f} ({gof['interpretation']})")
    print(f"RMSE: {gof['rmse']:.4f}")
    print(f"MAE: {gof['mae']:.4f}")
    print()
    
    # Example 3: Correlation matrix
    print("Example 3: Correlation Matrix")
    data = {
        'height': [160, 165, 170, 175, 180, 185],
        'weight': [50, 55, 65, 70, 80, 85],
        'age': [20, 25, 30, 35, 40, 45],
        'income': [30000, 35000, 45000, 50000, 60000, 65000]
    }
    
    corr_matrix = analyzer.correlation_matrix(data)
    print("Correlation Matrix:")
    print(corr_matrix.round(3))


if __name__ == "__main__":
    print("Running Bivariate Analysis Tests...\n")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*60 + "\n")
    
    # Run interactive examples
    run_interactive_examples()
