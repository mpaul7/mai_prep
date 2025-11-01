"""
Outlier Detection and Analysis for Data Science Interview Preparation

This module covers comprehensive outlier detection methods commonly tested in 
data science interviews, particularly HackerRank-style problems.

Topics covered:
1. Statistical methods for outlier detection (Z-score, IQR, Modified Z-score)
2. Distance-based outlier detection
3. Isolation Forest and other ML-based methods
4. Outlier impact analysis
5. Outlier handling strategies
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional
import statistics
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some advanced features may not work.")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML-based outlier detection methods will not work.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization features will not work.")


class OutlierDetector:
    """
    A comprehensive toolkit for outlier detection and analysis.
    Designed for data science interview preparation.
    """
    
    def __init__(self):
        """Initialize the outlier detector."""
        pass
    
    def z_score_outliers(self, data: List[Union[int, float]], 
                        threshold: float = 3.0) -> Dict[str, Union[List, float]]:
        """
        Detect outliers using Z-score method.
        
        Z-score = (x - mean) / standard_deviation
        Typically, |Z-score| > 3 indicates an outlier.
        
        Args:
            data: List of numeric values
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Dictionary containing outliers, indices, and statistics
            
        Example:
            >>> detector = OutlierDetector()
            >>> data = [1, 2, 3, 4, 5, 100]
            >>> result = detector.z_score_outliers(data)
            >>> print(result['outliers'])  # [100]
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for Z-score calculation")
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array, ddof=1)  # Sample standard deviation
        
        if std == 0:
            return {
                'outliers': [],
                'outlier_indices': [],
                'z_scores': [0.0] * len(data),
                'threshold': threshold,
                'mean': mean,
                'std': std
            }
        
        z_scores = np.abs((data_array - mean) / std)
        outlier_mask = z_scores > threshold
        
        return {
            'outliers': data_array[outlier_mask].tolist(),
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'z_scores': z_scores.tolist(),
            'threshold': threshold,
            'mean': mean,
            'std': std
        }
    
    def modified_z_score_outliers(self, data: List[Union[int, float]], 
                                 threshold: float = 3.5) -> Dict[str, Union[List, float]]:
        """
        Detect outliers using Modified Z-score method (more robust).
        
        Modified Z-score = 0.6745 * (x - median) / MAD
        Where MAD is the Median Absolute Deviation.
        
        Args:
            data: List of numeric values
            threshold: Modified Z-score threshold for outlier detection
            
        Returns:
            Dictionary containing outliers, indices, and statistics
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for Modified Z-score calculation")
        
        data_array = np.array(data)
        median = np.median(data_array)
        mad = np.median(np.abs(data_array - median))
        
        if mad == 0:
            return {
                'outliers': [],
                'outlier_indices': [],
                'modified_z_scores': [0.0] * len(data),
                'threshold': threshold,
                'median': median,
                'mad': mad
            }
        
        modified_z_scores = 0.6745 * (data_array - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        return {
            'outliers': data_array[outlier_mask].tolist(),
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'modified_z_scores': modified_z_scores.tolist(),
            'threshold': threshold,
            'median': median,
            'mad': mad
        }
    
    def iqr_outliers(self, data: List[Union[int, float]], 
                    k: float = 1.5) -> Dict[str, Union[List, float]]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Outliers are values below Q1 - k*IQR or above Q3 + k*IQR
        where Q1 is 25th percentile, Q3 is 75th percentile, IQR = Q3 - Q1
        
        Args:
            data: List of numeric values
            k: IQR multiplier (typically 1.5 for outliers, 3.0 for extreme outliers)
            
        Returns:
            Dictionary containing outliers, indices, and statistics
        """
        if len(data) < 4:
            raise ValueError("Need at least 4 data points for IQR calculation")
        
        data_array = np.array(data)
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        outlier_mask = (data_array < lower_bound) | (data_array > upper_bound)
        
        return {
            'outliers': data_array[outlier_mask].tolist(),
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'k': k
        }
    
    def isolation_forest_outliers(self, data: List[Union[int, float]], 
                                 contamination: float = 0.1) -> Dict[str, Union[List, np.ndarray]]:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Args:
            data: List of numeric values
            contamination: Expected proportion of outliers in the data
            
        Returns:
            Dictionary containing outliers, indices, and anomaly scores
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Isolation Forest. Install with: pip install scikit-learn")
        
        if len(data) < 10:
            raise ValueError("Need at least 10 data points for Isolation Forest")
        
        data_array = np.array(data).reshape(-1, 1)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data_array)
        anomaly_scores = iso_forest.score_samples(data_array)
        
        outlier_mask = outlier_labels == -1
        
        return {
            'outliers': np.array(data)[outlier_mask].tolist(),
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'outlier_labels': outlier_labels.tolist(),
            'contamination': contamination
        }
    
    def local_outlier_factor(self, data: List[Union[int, float]], 
                           n_neighbors: int = 20) -> Dict[str, Union[List, np.ndarray]]:
        """
        Detect outliers using Local Outlier Factor (LOF).
        
        Args:
            data: List of numeric values
            n_neighbors: Number of neighbors to consider
            
        Returns:
            Dictionary containing outliers, indices, and LOF scores
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for LOF. Install with: pip install scikit-learn")
        
        if len(data) < n_neighbors + 1:
            raise ValueError(f"Need at least {n_neighbors + 1} data points for LOF")
        
        data_array = np.array(data).reshape(-1, 1)
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        outlier_labels = lof.fit_predict(data_array)
        lof_scores = -lof.negative_outlier_factor_
        
        outlier_mask = outlier_labels == -1
        
        return {
            'outliers': np.array(data)[outlier_mask].tolist(),
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'lof_scores': lof_scores.tolist(),
            'outlier_labels': outlier_labels.tolist(),
            'n_neighbors': n_neighbors
        }
    
    def percentile_outliers(self, data: List[Union[int, float]], 
                          lower_percentile: float = 5, 
                          upper_percentile: float = 95) -> Dict[str, Union[List, float]]:
        """
        Detect outliers using percentile method.
        
        Args:
            data: List of numeric values
            lower_percentile: Lower percentile threshold
            upper_percentile: Upper percentile threshold
            
        Returns:
            Dictionary containing outliers, indices, and thresholds
        """
        data_array = np.array(data)
        lower_threshold = np.percentile(data_array, lower_percentile)
        upper_threshold = np.percentile(data_array, upper_percentile)
        
        outlier_mask = (data_array < lower_threshold) | (data_array > upper_threshold)
        
        return {
            'outliers': data_array[outlier_mask].tolist(),
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'lower_percentile': lower_percentile,
            'upper_percentile': upper_percentile
        }
    
    def compare_methods(self, data: List[Union[int, float]]) -> pd.DataFrame:
        """
        Compare different outlier detection methods on the same data.
        
        Args:
            data: List of numeric values
            
        Returns:
            DataFrame comparing results from different methods
        """
        results = []
        
        try:
            z_result = self.z_score_outliers(data)
            results.append({
                'method': 'Z-Score',
                'outliers_count': len(z_result['outliers']),
                'outliers': z_result['outliers'],
                'indices': z_result['outlier_indices']
            })
        except Exception as e:
            results.append({
                'method': 'Z-Score',
                'outliers_count': 'Error',
                'outliers': str(e),
                'indices': []
            })
        
        try:
            mod_z_result = self.modified_z_score_outliers(data)
            results.append({
                'method': 'Modified Z-Score',
                'outliers_count': len(mod_z_result['outliers']),
                'outliers': mod_z_result['outliers'],
                'indices': mod_z_result['outlier_indices']
            })
        except Exception as e:
            results.append({
                'method': 'Modified Z-Score',
                'outliers_count': 'Error',
                'outliers': str(e),
                'indices': []
            })
        
        try:
            iqr_result = self.iqr_outliers(data)
            results.append({
                'method': 'IQR',
                'outliers_count': len(iqr_result['outliers']),
                'outliers': iqr_result['outliers'],
                'indices': iqr_result['outlier_indices']
            })
        except Exception as e:
            results.append({
                'method': 'IQR',
                'outliers_count': 'Error',
                'outliers': str(e),
                'indices': []
            })
        
        try:
            percentile_result = self.percentile_outliers(data)
            results.append({
                'method': 'Percentile (5-95)',
                'outliers_count': len(percentile_result['outliers']),
                'outliers': percentile_result['outliers'],
                'indices': percentile_result['outlier_indices']
            })
        except Exception as e:
            results.append({
                'method': 'Percentile (5-95)',
                'outliers_count': 'Error',
                'outliers': str(e),
                'indices': []
            })
        
        if len(data) >= 10 and SKLEARN_AVAILABLE:
            try:
                iso_result = self.isolation_forest_outliers(data)
                results.append({
                    'method': 'Isolation Forest',
                    'outliers_count': len(iso_result['outliers']),
                    'outliers': iso_result['outliers'],
                    'indices': iso_result['outlier_indices']
                })
            except Exception as e:
                results.append({
                    'method': 'Isolation Forest',
                    'outliers_count': 'Error',
                    'outliers': str(e),
                    'indices': []
                })
        
        return pd.DataFrame(results)
    
    def analyze_outlier_impact(self, data: List[Union[int, float]], 
                             outlier_indices: List[int]) -> Dict[str, float]:
        """
        Analyze the impact of outliers on statistical measures.
        
        Args:
            data: Original data
            outlier_indices: Indices of detected outliers
            
        Returns:
            Dictionary with impact analysis
        """
        data_array = np.array(data)
        clean_data = np.delete(data_array, outlier_indices)
        
        if len(clean_data) == 0:
            raise ValueError("Cannot analyze impact - all data points are outliers")
        
        return {
            'original_mean': float(np.mean(data_array)),
            'clean_mean': float(np.mean(clean_data)),
            'mean_change': float(np.mean(data_array) - np.mean(clean_data)),
            'original_std': float(np.std(data_array, ddof=1)),
            'clean_std': float(np.std(clean_data, ddof=1)),
            'std_change': float(np.std(data_array, ddof=1) - np.std(clean_data, ddof=1)),
            'original_median': float(np.median(data_array)),
            'clean_median': float(np.median(clean_data)),
            'median_change': float(np.median(data_array) - np.median(clean_data)),
            'outliers_removed': len(outlier_indices),
            'data_points_remaining': len(clean_data)
        }


def hackerrank_outlier_problems():
    """
    Collection of HackerRank-style problems for outlier detection.
    """
    
    def problem_1_basic_outlier_detection():
        """
        Problem 1: Basic Outlier Detection using Z-score
        
        Given a list of numbers, identify outliers using Z-score method.
        Return the count of outliers with |Z-score| > 2.
        
        Input: [1, 2, 3, 4, 5, 100, 2, 3, 4, 5]
        Expected Output: 1
        """
        def count_z_score_outliers(data: List[Union[int, float]], threshold: float = 2.0) -> int:
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
        
        # Test case
        test_data = [1, 2, 3, 4, 5, 100, 2, 3, 4, 5]
        result = count_z_score_outliers(test_data)
        print(f"Problem 1 - Input: {test_data}")
        print(f"Problem 1 - Z-score outliers (threshold=2): {result}")
        return result
    
    def problem_2_iqr_outlier_detection():
        """
        Problem 2: IQR Outlier Detection
        
        Given a list of numbers, identify outliers using IQR method.
        Return the outlier values.
        
        Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        """
        def find_iqr_outliers(data: List[Union[int, float]], k: float = 1.5) -> List[Union[int, float]]:
            if len(data) < 4:
                return []
            
            sorted_data = sorted(data)
            n = len(sorted_data)
            
            # Calculate Q1 and Q3
            q1_index = n // 4
            q3_index = 3 * n // 4
            
            q1 = sorted_data[q1_index]
            q3 = sorted_data[q3_index]
            iqr = q3 - q1
            
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            
            outliers = [x for x in data if x < lower_bound or x > upper_bound]
            return outliers
        
        # Test case
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        result = find_iqr_outliers(test_data)
        print(f"Problem 2 - Input: {test_data}")
        print(f"Problem 2 - IQR outliers: {result}")
        return result
    
    def problem_3_outlier_impact_on_mean():
        """
        Problem 3: Calculate Impact of Outliers on Mean
        
        Given a dataset, calculate how much the mean changes when outliers are removed.
        Use Z-score method with threshold 2.5 to identify outliers.
        
        Input: [10, 12, 14, 16, 18, 20, 22, 24, 26, 200]
        """
        def calculate_mean_impact(data: List[Union[int, float]], z_threshold: float = 2.5) -> float:
            if len(data) < 2:
                return 0.0
            
            # Calculate original mean
            original_mean = sum(data) / len(data)
            
            # Find outliers using Z-score
            mean = original_mean
            variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
            std = variance ** 0.5
            
            if std == 0:
                return 0.0
            
            clean_data = []
            for x in data:
                z_score = abs((x - mean) / std)
                if z_score <= z_threshold:
                    clean_data.append(x)
            
            if len(clean_data) == 0:
                return 0.0
            
            clean_mean = sum(clean_data) / len(clean_data)
            impact = abs(original_mean - clean_mean)
            
            return round(impact, 2)
        
        # Test case
        test_data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 200]
        result = calculate_mean_impact(test_data)
        original_mean = sum(test_data) / len(test_data)
        print(f"Problem 3 - Input: {test_data}")
        print(f"Problem 3 - Original mean: {original_mean:.2f}")
        print(f"Problem 3 - Mean impact after outlier removal: {result}")
        return result
    
    def problem_4_robust_vs_non_robust_statistics():
        """
        Problem 4: Compare Robust vs Non-Robust Statistics
        
        Given a dataset with outliers, compare the difference between
        mean vs median and standard deviation vs MAD.
        
        Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]
        """
        def compare_robust_statistics(data: List[Union[int, float]]) -> Dict[str, float]:
            # Non-robust statistics
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
            std = variance ** 0.5
            
            # Robust statistics
            sorted_data = sorted(data)
            n = len(sorted_data)
            if n % 2 == 0:
                median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
            else:
                median = sorted_data[n//2]
            
            # MAD (Median Absolute Deviation)
            mad = sorted([abs(x - median) for x in data])[n//2]
            
            return {
                'mean': round(mean, 2),
                'median': round(median, 2),
                'mean_median_diff': round(abs(mean - median), 2),
                'std': round(std, 2),
                'mad': round(mad, 2),
                'std_mad_ratio': round(std / mad if mad != 0 else float('inf'), 2)
            }
        
        # Test case
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]
        result = compare_robust_statistics(test_data)
        print(f"Problem 4 - Input: {test_data}")
        print(f"Problem 4 - Statistics comparison: {result}")
        return result
    
    def problem_5_outlier_percentage():
        """
        Problem 5: Calculate Outlier Percentage
        
        Given a dataset, calculate what percentage of data points are outliers
        using the IQR method.
        
        Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300]
        """
        def calculate_outlier_percentage(data: List[Union[int, float]], k: float = 1.5) -> float:
            if len(data) < 4:
                return 0.0
            
            sorted_data = sorted(data)
            n = len(sorted_data)
            
            # Calculate Q1 and Q3
            q1 = sorted_data[n//4]
            q3 = sorted_data[3*n//4]
            iqr = q3 - q1
            
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            
            outlier_count = sum(1 for x in data if x < lower_bound or x > upper_bound)
            percentage = (outlier_count / len(data)) * 100
            
            return round(percentage, 2)
        
        # Test case
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300]
        result = calculate_outlier_percentage(test_data)
        print(f"Problem 5 - Input: {test_data}")
        print(f"Problem 5 - Outlier percentage: {result}%")
        return result
    
    # Run all problems
    print("=== HackerRank Style Outlier Problems ===\n")
    problem_1_basic_outlier_detection()
    print()
    problem_2_iqr_outlier_detection()
    print()
    problem_3_outlier_impact_on_mean()
    print()
    problem_4_robust_vs_non_robust_statistics()
    print()
    problem_5_outlier_percentage()


if __name__ == "__main__":
    # Demonstrate the outlier detection toolkit
    print("=== Outlier Detection Toolkit Demonstration ===\n")
    
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 100).tolist()
    print(f"Normal data: {len(normal_data)}")
    outliers = [150, 200, -50]
    sample_data = normal_data + outliers
    
    print(f"Sample data size: {len(sample_data)}")
    print(f"Added outliers: {outliers}")
    print(f"Data range: {min(sample_data):.2f} to {max(sample_data):.2f}")
    print()
    
    # Initialize detector
    detector = OutlierDetector()
    
    # Compare different methods
    print("Comparing outlier detection methods:")
    comparison = detector.compare_methods(sample_data)
    print(comparison.to_string(index=False))
    print()
    
    # Detailed analysis using IQR method
    iqr_result = detector.iqr_outliers(sample_data)
    print(f"IQR Method Details:")
    print(f"Q1: {iqr_result['q1']:.2f}, Q3: {iqr_result['q3']:.2f}")
    print(f"IQR: {iqr_result['iqr']:.2f}")
    print(f"Bounds: [{iqr_result['lower_bound']:.2f}, {iqr_result['upper_bound']:.2f}]")
    print(f"Detected outliers: {iqr_result['outliers']}")
    print()
    
    # Impact analysis
    impact = detector.analyze_outlier_impact(sample_data, iqr_result['outlier_indices'])
    print("Outlier Impact Analysis:")
    for key, value in impact.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*60 + "\n")
    
    # Run HackerRank-style problems
    hackerrank_outlier_problems()

