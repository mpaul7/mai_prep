"""
Dispersion Metrics for Data Science Interview Preparation

This module covers comprehensive dispersion metrics commonly tested in 
data science interviews, particularly HackerRank-style problems.

Topics covered:
1. Range (Min-Max difference)
2. Interquartile Range (IQR)
3. Variance (Population and Sample)
4. Standard Deviation (Population and Sample)
5. Coefficient of Variation
6. Mean Absolute Deviation (MAD)
7. Robust vs Non-robust measures
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import statistics
import math
from collections import Counter


class DispersionCalculator:
    """
    A comprehensive toolkit for calculating dispersion metrics.
    Designed for data science interview preparation.
    """
    
    def __init__(self):
        """Initialize the dispersion calculator."""
        pass
    
    def calculate_range(self, data: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate the range (max - min) of the data.
        
        Args:
            data: List of numeric values
            
        Returns:
            Dictionary containing range statistics
            
        Example:
            >>> calc = DispersionCalculator()
            >>> data = [1, 2, 3, 4, 5]
            >>> result = calc.calculate_range(data)
            >>> print(result['range'])  # 4.0
        """
        if not data:
            raise ValueError("Cannot calculate range of empty data")
        
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val
        
        return {
            'range': float(range_val),
            'min': float(min_val),
            'max': float(max_val),
            'midrange': float((min_val + max_val) / 2)
        }
    
    def calculate_iqr(self, data: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate the Interquartile Range (IQR) and related quartile statistics.
        
        IQR = Q3 - Q1 (75th percentile - 25th percentile)
        
        Args:
            data: List of numeric values
            
        Returns:
            Dictionary containing IQR and quartile statistics
            
        Example:
            >>> calc = DispersionCalculator()
            >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> result = calc.calculate_iqr(data)
            >>> print(result['iqr'])  # 4.5
        """
        if len(data) < 4:
            raise ValueError("Need at least 4 data points for IQR calculation")
        
        data_array = np.array(data)
        q1 = np.percentile(data_array, 25)
        q2 = np.percentile(data_array, 50)  # Median
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        
        return {
            'iqr': float(iqr),
            'q1': float(q1),
            'q2_median': float(q2),
            'q3': float(q3),
            'lower_fence': float(q1 - 1.5 * iqr),  # Outlier detection boundary
            'upper_fence': float(q3 + 1.5 * iqr),  # Outlier detection boundary
            'quartile_coefficient_dispersion': float(iqr / (q1 + q3)) if (q1 + q3) != 0 else 0.0
        }
    
    def calculate_variance(self, data: List[Union[int, float]], 
                          sample: bool = True) -> Dict[str, float]:
        """
        Calculate variance (both sample and population).
        
        Sample variance: Σ(x - x̄)² / (n - 1)
        Population variance: Σ(x - μ)² / n
        
        Args:
            data: List of numeric values
            sample: If True, calculate sample variance; if False, population variance
            
        Returns:
            Dictionary containing variance statistics
            
        Example:
            >>> calc = DispersionCalculator()
            >>> data = [1, 2, 3, 4, 5]
            >>> result = calc.calculate_variance(data, sample=True)
            >>> print(result['variance'])  # 2.5
        """
        if not data:
            raise ValueError("Cannot calculate variance of empty data")
        
        if len(data) == 1:
            if sample:
                raise ValueError("Cannot calculate sample variance with only one data point")
            else:
                return {
                    'variance': 0.0,
                    'mean': float(data[0]),
                    'n': 1,
                    'type': 'population'
                }
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        
        if sample:
            variance = np.var(data_array, ddof=1)  # Sample variance
            variance_type = 'sample'
            denominator = len(data) - 1
        else:
            variance = np.var(data_array, ddof=0)  # Population variance
            variance_type = 'population'
            denominator = len(data)
        
        return {
            'variance': float(variance),
            'mean': float(mean),
            'n': len(data),
            'type': variance_type,
            'denominator': denominator,
            'sum_squared_deviations': float(variance * denominator)
        }
    
    def calculate_standard_deviation(self, data: List[Union[int, float]], 
                                   sample: bool = True) -> Dict[str, float]:
        """
        Calculate standard deviation (both sample and population).
        
        Standard deviation is the square root of variance.
        
        Args:
            data: List of numeric values
            sample: If True, calculate sample std dev; if False, population std dev
            
        Returns:
            Dictionary containing standard deviation statistics
            
        Example:
            >>> calc = DispersionCalculator()
            >>> data = [1, 2, 3, 4, 5]
            >>> result = calc.calculate_standard_deviation(data, sample=True)
            >>> print(result['std_dev'])  # 1.58...
        """
        variance_result = self.calculate_variance(data, sample=sample)
        std_dev = math.sqrt(variance_result['variance'])
        
        result = variance_result.copy()
        result['std_dev'] = float(std_dev)
        
        return result
    
    def calculate_coefficient_of_variation(self, data: List[Union[int, float]], 
                                         sample: bool = True) -> Dict[str, float]:
        """
        Calculate the Coefficient of Variation (CV).
        
        CV = (Standard Deviation / Mean) × 100%
        Used to compare variability between datasets with different means.
        
        Args:
            data: List of numeric values
            sample: If True, use sample std dev; if False, use population std dev
            
        Returns:
            Dictionary containing CV and related statistics
            
        Example:
            >>> calc = DispersionCalculator()
            >>> data = [10, 20, 30, 40, 50]
            >>> result = calc.calculate_coefficient_of_variation(data)
            >>> print(result['cv_percent'])  # CV as percentage
        """
        std_result = self.calculate_standard_deviation(data, sample=sample)
        mean = std_result['mean']
        std_dev = std_result['std_dev']
        
        if mean == 0:
            raise ValueError("Cannot calculate coefficient of variation when mean is zero")
        
        cv = std_dev / abs(mean)
        cv_percent = cv * 100
        
        result = std_result.copy()
        result.update({
            'cv': float(cv),
            'cv_percent': float(cv_percent),
            'interpretation': self._interpret_cv(cv_percent)
        })
        
        return result
    
    def calculate_mean_absolute_deviation(self, data: List[Union[int, float]], 
                                        center: str = 'mean') -> Dict[str, float]:
        """
        Calculate Mean Absolute Deviation (MAD).
        
        MAD from mean: Σ|x - x̄| / n
        MAD from median: Σ|x - median| / n (more robust)
        
        Args:
            data: List of numeric values
            center: 'mean' or 'median' - what to calculate deviations from
            
        Returns:
            Dictionary containing MAD statistics
            
        Example:
            >>> calc = DispersionCalculator()
            >>> data = [1, 2, 3, 4, 5]
            >>> result = calc.calculate_mean_absolute_deviation(data, 'mean')
            >>> print(result['mad'])  # 1.2
        """
        if not data:
            raise ValueError("Cannot calculate MAD of empty data")
        
        data_array = np.array(data)
        
        if center == 'mean':
            center_value = np.mean(data_array)
        elif center == 'median':
            center_value = np.median(data_array)
        else:
            raise ValueError("Center must be 'mean' or 'median'")
        
        absolute_deviations = np.abs(data_array - center_value)
        mad = np.mean(absolute_deviations)
        
        return {
            'mad': float(mad),
            'center_type': center,
            'center_value': float(center_value),
            'n': len(data),
            'sum_absolute_deviations': float(np.sum(absolute_deviations))
        }
    
    def _interpret_cv(self, cv_percent: float) -> str:
        """Helper method to interpret coefficient of variation."""
        if cv_percent < 15:
            return "Low variability"
        elif cv_percent < 35:
            return "Moderate variability"
        else:
            return "High variability"
    
    def compare_dispersion_measures(self, data: List[Union[int, float]]) -> pd.DataFrame:
        """
        Calculate and compare all dispersion measures for the given data.
        
        Args:
            data: List of numeric values
            
        Returns:
            DataFrame with all dispersion measures
        """
        results = []
        
        try:
            range_result = self.calculate_range(data)
            results.append({
                'measure': 'Range',
                'value': range_result['range'],
                'interpretation': f"Spread from {range_result['min']} to {range_result['max']}",
                'robust': False
            })
        except Exception as e:
            results.append({
                'measure': 'Range',
                'value': f'Error: {str(e)}',
                'interpretation': 'N/A',
                'robust': False
            })
        
        try:
            iqr_result = self.calculate_iqr(data)
            results.append({
                'measure': 'IQR',
                'value': iqr_result['iqr'],
                'interpretation': f"Middle 50% spread: {iqr_result['iqr']:.2f}",
                'robust': True
            })
        except Exception as e:
            results.append({
                'measure': 'IQR',
                'value': f'Error: {str(e)}',
                'interpretation': 'N/A',
                'robust': True
            })
        
        try:
            var_result = self.calculate_variance(data, sample=True)
            results.append({
                'measure': 'Sample Variance',
                'value': var_result['variance'],
                'interpretation': f"Average squared deviation: {var_result['variance']:.2f}",
                'robust': False
            })
        except Exception as e:
            results.append({
                'measure': 'Sample Variance',
                'value': f'Error: {str(e)}',
                'interpretation': 'N/A',
                'robust': False
            })
        
        try:
            std_result = self.calculate_standard_deviation(data, sample=True)
            results.append({
                'measure': 'Sample Std Dev',
                'value': std_result['std_dev'],
                'interpretation': f"Typical deviation: {std_result['std_dev']:.2f}",
                'robust': False
            })
        except Exception as e:
            results.append({
                'measure': 'Sample Std Dev',
                'value': f'Error: {str(e)}',
                'interpretation': 'N/A',
                'robust': False
            })
        
        try:
            cv_result = self.calculate_coefficient_of_variation(data, sample=True)
            results.append({
                'measure': 'Coefficient of Variation',
                'value': f"{cv_result['cv_percent']:.2f}%",
                'interpretation': cv_result['interpretation'],
                'robust': False
            })
        except Exception as e:
            results.append({
                'measure': 'Coefficient of Variation',
                'value': f'Error: {str(e)}',
                'interpretation': 'N/A',
                'robust': False
            })
        
        try:
            mad_mean_result = self.calculate_mean_absolute_deviation(data, 'mean')
            results.append({
                'measure': 'MAD (from mean)',
                'value': mad_mean_result['mad'],
                'interpretation': f"Average absolute deviation: {mad_mean_result['mad']:.2f}",
                'robust': False
            })
        except Exception as e:
            results.append({
                'measure': 'MAD (from mean)',
                'value': f'Error: {str(e)}',
                'interpretation': 'N/A',
                'robust': False
            })
        
        try:
            mad_median_result = self.calculate_mean_absolute_deviation(data, 'median')
            results.append({
                'measure': 'MAD (from median)',
                'value': mad_median_result['mad'],
                'interpretation': f"Robust average deviation: {mad_median_result['mad']:.2f}",
                'robust': True
            })
        except Exception as e:
            results.append({
                'measure': 'MAD (from median)',
                'value': f'Error: {str(e)}',
                'interpretation': 'N/A',
                'robust': True
            })
        
        return pd.DataFrame(results)
    
    def analyze_dispersion_impact(self, data1: List[Union[int, float]], 
                                data2: List[Union[int, float]]) -> Dict[str, Dict]:
        """
        Compare dispersion measures between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Dictionary comparing dispersion measures
        """
        comparison = {}
        
        # Calculate measures for both datasets
        measures1 = self.compare_dispersion_measures(data1)
        measures2 = self.compare_dispersion_measures(data2)
        
        # Compare each measure
        for i, measure in enumerate(measures1['measure']):
            val1 = measures1.iloc[i]['value']
            val2 = measures2.iloc[i]['value']
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                difference = val2 - val1
                percent_change = ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf')
                
                comparison[measure] = {
                    'dataset1': val1,
                    'dataset2': val2,
                    'difference': difference,
                    'percent_change': percent_change,
                    'higher_dispersion': 'Dataset 2' if val2 > val1 else 'Dataset 1' if val1 > val2 else 'Equal'
                }
            else:
                comparison[measure] = {
                    'dataset1': val1,
                    'dataset2': val2,
                    'difference': 'Cannot calculate',
                    'percent_change': 'Cannot calculate',
                    'higher_dispersion': 'Cannot determine'
                }
        
        return comparison


def hackerrank_dispersion_problems():
    """
    Collection of HackerRank-style problems for dispersion metrics.
    """
    
    def problem_1_basic_range():
        """
        Problem 1: Calculate Range
        
        Given a list of numbers, calculate the range (max - min).
        
        Input: [10, 5, 8, 12, 3, 15, 7]
        Expected Output: 12
        """
        def calculate_range(data: List[Union[int, float]]) -> float:
            if not data:
                return 0
            return max(data) - min(data)
        
        # Test case
        test_data = [10, 5, 8, 12, 3, 15, 7]
        result = calculate_range(test_data)
        print(f"Problem 1 - Input: {test_data}")
        print(f"Problem 1 - Range: {result}")
        return result
    
    def problem_2_iqr_calculation():
        """
        Problem 2: Calculate IQR
        
        Given a list of numbers, calculate the Interquartile Range.
        
        Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Expected Output: 4.5
        """
        def calculate_iqr(data: List[Union[int, float]]) -> float:
            if len(data) < 4:
                return 0
            
            sorted_data = sorted(data)
            n = len(sorted_data)
            
            # Calculate Q1 and Q3 positions
            q1_pos = (n + 1) * 0.25
            q3_pos = (n + 1) * 0.75
            
            # Linear interpolation for quartiles
            def get_quartile(pos):
                if pos == int(pos):
                    return sorted_data[int(pos) - 1]
                else:
                    lower = int(pos) - 1
                    upper = int(pos)
                    weight = pos - int(pos)
                    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
            
            q1 = get_quartile(q1_pos)
            q3 = get_quartile(q3_pos)
            
            return q3 - q1
        
        # Test case
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = calculate_iqr(test_data)
        print(f"Problem 2 - Input: {test_data}")
        print(f"Problem 2 - IQR: {result}")
        return result
    
    def problem_3_sample_vs_population_variance():
        """
        Problem 3: Sample vs Population Variance
        
        Given a dataset, calculate both sample and population variance.
        Return the difference between them.
        
        Input: [2, 4, 6, 8, 10]
        """
        def variance_difference(data: List[Union[int, float]]) -> float:
            if len(data) <= 1:
                return 0
            
            mean = sum(data) / len(data)
            sum_squared_diff = sum((x - mean) ** 2 for x in data)
            
            population_var = sum_squared_diff / len(data)
            sample_var = sum_squared_diff / (len(data) - 1)
            
            return sample_var - population_var
        
        # Test case
        test_data = [2, 4, 6, 8, 10]
        result = variance_difference(test_data)
        
        # Calculate individual variances for display
        mean = sum(test_data) / len(test_data)
        sum_sq_diff = sum((x - mean) ** 2 for x in test_data)
        pop_var = sum_sq_diff / len(test_data)
        sample_var = sum_sq_diff / (len(test_data) - 1)
        
        print(f"Problem 3 - Input: {test_data}")
        print(f"Problem 3 - Population Variance: {pop_var:.2f}")
        print(f"Problem 3 - Sample Variance: {sample_var:.2f}")
        print(f"Problem 3 - Difference: {result:.2f}")
        return result
    
    def problem_4_coefficient_of_variation():
        """
        Problem 4: Coefficient of Variation Comparison
        
        Given two datasets, determine which has higher relative variability
        using the coefficient of variation.
        
        Dataset 1: [100, 110, 120, 130, 140]
        Dataset 2: [10, 11, 12, 13, 14]
        """
        def compare_cv(data1: List[Union[int, float]], 
                      data2: List[Union[int, float]]) -> int:
            def calculate_cv(data):
                if not data or len(data) < 2:
                    return 0
                
                mean = sum(data) / len(data)
                if mean == 0:
                    return float('inf')
                
                variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
                std_dev = variance ** 0.5
                
                return (std_dev / abs(mean)) * 100
            
            cv1 = calculate_cv(data1)
            cv2 = calculate_cv(data2)
            
            if cv1 > cv2:
                return 1
            elif cv2 > cv1:
                return 2
            else:
                return 0
        
        # Test case
        dataset1 = [100, 110, 120, 130, 140]
        dataset2 = [10, 11, 12, 13, 14]
        result = compare_cv(dataset1, dataset2)
        
        # Calculate CVs for display
        def calc_cv_display(data):
            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
            std = var ** 0.5
            return (std / mean) * 100
        
        cv1 = calc_cv_display(dataset1)
        cv2 = calc_cv_display(dataset2)
        
        print(f"Problem 4 - Dataset 1: {dataset1} (CV: {cv1:.2f}%)")
        print(f"Problem 4 - Dataset 2: {dataset2} (CV: {cv2:.2f}%)")
        print(f"Problem 4 - Higher CV: Dataset {result if result != 0 else 'Equal'}")
        return result
    
    def problem_5_robust_vs_nonrobust():
        """
        Problem 5: Robust vs Non-robust Measures
        
        Given a dataset with an outlier, compare how much the standard deviation
        changes vs how much the IQR changes when the outlier is removed.
        
        Input: [1, 2, 3, 4, 5, 100]
        """
        def measure_robustness(data: List[Union[int, float]]) -> Dict[str, float]:
            if len(data) < 4:
                return {'std_change': 0, 'iqr_change': 0}
            
            # Original measures
            def calc_std(data_list):
                if len(data_list) < 2:
                    return 0
                mean = sum(data_list) / len(data_list)
                var = sum((x - mean) ** 2 for x in data_list) / (len(data_list) - 1)
                return var ** 0.5
            
            def calc_iqr(data_list):
                if len(data_list) < 4:
                    return 0
                sorted_data = sorted(data_list)
                n = len(sorted_data)
                q1_idx = n // 4
                q3_idx = 3 * n // 4
                return sorted_data[q3_idx] - sorted_data[q1_idx]
            
            original_std = calc_std(data)
            original_iqr = calc_iqr(data)
            
            # Remove potential outlier (assume it's the max value)
            data_no_outlier = [x for x in data if x != max(data)]
            
            new_std = calc_std(data_no_outlier)
            new_iqr = calc_iqr(data_no_outlier)
            
            std_change = abs(original_std - new_std)
            iqr_change = abs(original_iqr - new_iqr)
            
            return {
                'original_std': original_std,
                'new_std': new_std,
                'std_change': std_change,
                'original_iqr': original_iqr,
                'new_iqr': new_iqr,
                'iqr_change': iqr_change
            }
        
        # Test case
        test_data = [1, 2, 3, 4, 5, 100]
        result = measure_robustness(test_data)
        
        print(f"Problem 5 - Input: {test_data}")
        print(f"Problem 5 - Original Std Dev: {result['original_std']:.2f}")
        print(f"Problem 5 - New Std Dev: {result['new_std']:.2f}")
        print(f"Problem 5 - Std Dev Change: {result['std_change']:.2f}")
        print(f"Problem 5 - Original IQR: {result['original_iqr']:.2f}")
        print(f"Problem 5 - New IQR: {result['new_iqr']:.2f}")
        print(f"Problem 5 - IQR Change: {result['iqr_change']:.2f}")
        return result
    
    # Run all problems
    print("=== HackerRank Style Dispersion Problems ===\n")
    problem_1_basic_range()
    print()
    problem_2_iqr_calculation()
    print()
    problem_3_sample_vs_population_variance()
    print()
    problem_4_coefficient_of_variation()
    print()
    problem_5_robust_vs_nonrobust()


if __name__ == "__main__":
    # Demonstrate the dispersion calculator
    print("=== Dispersion Metrics Toolkit Demonstration ===\n")
    
    # Create sample datasets
    np.random.seed(42)
    normal_data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    skewed_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50]
    
    print(f"Normal data: {normal_data}")
    print(f"Skewed data: {skewed_data}")
    print()
    
    # Initialize calculator
    calc = DispersionCalculator()
    
    # Compare dispersion measures for normal data
    print("Dispersion measures for normal data:")
    normal_comparison = calc.compare_dispersion_measures(normal_data)
    print(normal_comparison.to_string(index=False))
    print()
    
    # Compare dispersion measures for skewed data
    print("Dispersion measures for skewed data:")
    skewed_comparison = calc.compare_dispersion_measures(skewed_data)
    print(skewed_comparison.to_string(index=False))
    print()
    
    # Analyze impact of skewness
    print("Impact analysis (Normal vs Skewed):")
    impact = calc.analyze_dispersion_impact(normal_data, skewed_data)
    for measure, stats in impact.items():
        if isinstance(stats['dataset1'], (int, float)):
            print(f"{measure}:")
            print(f"  Normal: {stats['dataset1']:.2f}")
            print(f"  Skewed: {stats['dataset2']:.2f}")
            print(f"  Difference: {stats['difference']:.2f}")
            print(f"  Higher dispersion: {stats['higher_dispersion']}")
            print()
    
    print("\n" + "="*60 + "\n")
    
    # Run HackerRank-style problems
    hackerrank_dispersion_problems()
