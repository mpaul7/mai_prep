"""
Error Metrics for Data Science Interview Preparation

This module covers comprehensive error metrics commonly tested in 
data science interviews, particularly HackerRank-style problems.

Topics covered:
1. Mean Squared Error (MSE)
2. Root Mean Squared Error (RMSE)
3. Mean Absolute Error (MAE)
4. Mean Absolute Percentage Error (MAPE)
5. Mean Absolute Scaled Error (MASE)
6. Symmetric Mean Absolute Percentage Error (SMAPE)
7. Mean Squared Logarithmic Error (MSLE)
8. R-squared and Adjusted R-squared
9. Mean Directional Accuracy (MDA)
10. Theil's U Statistic
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import math
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some advanced features may not work.")


class ErrorMetricsCalculator:
    """
    A comprehensive toolkit for calculating error metrics.
    Designed for data science interview preparation.
    """
    
    def __init__(self):
        """Initialize the error metrics calculator."""
        pass
    
    def mean_squared_error(self, y_true: List[Union[int, float]], 
                          y_pred: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate Mean Squared Error (MSE).
        
        MSE = (1/n) * Σ(y_true - y_pred)²
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing MSE and related statistics
            
        Example:
            >>> calc = ErrorMetricsCalculator()
            >>> y_true = [1, 2, 3, 4, 5]
            >>> y_pred = [1.1, 2.1, 2.9, 4.1, 4.9]
            >>> result = calc.mean_squared_error(y_true, y_pred)
            >>> print(result['mse'])  # 0.02
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate MSE for empty arrays")
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        squared_errors = (y_true_array - y_pred_array) ** 2
        mse = np.mean(squared_errors)
        
        return {
            'mse': float(mse),
            'sum_squared_errors': float(np.sum(squared_errors)),
            'n': len(y_true),
            'max_squared_error': float(np.max(squared_errors)),
            'min_squared_error': float(np.min(squared_errors)),
            'std_squared_errors': float(np.std(squared_errors))
        }
    
    def root_mean_squared_error(self, y_true: List[Union[int, float]], 
                               y_pred: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate Root Mean Squared Error (RMSE).
        
        RMSE = √(MSE) = √[(1/n) * Σ(y_true - y_pred)²]
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing RMSE and related statistics
        """
        mse_result = self.mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse_result['mse'])
        
        result = mse_result.copy()
        result['rmse'] = float(rmse)
        
        # Add normalized RMSE (NRMSE) - normalized by range
        y_true_array = np.array(y_true)
        y_range = np.max(y_true_array) - np.min(y_true_array)
        result['nrmse_range'] = float(rmse / y_range) if y_range != 0 else float('inf')
        
        # NRMSE normalized by mean
        y_mean = np.mean(y_true_array)
        result['nrmse_mean'] = float(rmse / y_mean) if y_mean != 0 else float('inf')
        
        return result
    
    def mean_absolute_error(self, y_true: List[Union[int, float]], 
                           y_pred: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate Mean Absolute Error (MAE).
        
        MAE = (1/n) * Σ|y_true - y_pred|
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing MAE and related statistics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate MAE for empty arrays")
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        absolute_errors = np.abs(y_true_array - y_pred_array)
        mae = np.mean(absolute_errors)
        
        return {
            'mae': float(mae),
            'sum_absolute_errors': float(np.sum(absolute_errors)),
            'n': len(y_true),
            'max_absolute_error': float(np.max(absolute_errors)),
            'min_absolute_error': float(np.min(absolute_errors)),
            'median_absolute_error': float(np.median(absolute_errors)),
            'std_absolute_errors': float(np.std(absolute_errors))
        }
    
    def mean_absolute_percentage_error(self, y_true: List[Union[int, float]], 
                                     y_pred: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing MAPE and related statistics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate MAPE for empty arrays")
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        # Handle zero values in y_true
        non_zero_mask = y_true_array != 0
        
        if not np.any(non_zero_mask):
            return {
                'mape': float('inf'),
                'n': len(y_true),
                'zero_count': len(y_true),
                'valid_count': 0,
                'interpretation': 'Cannot calculate MAPE: all actual values are zero'
            }
        
        # Calculate MAPE only for non-zero actual values
        valid_y_true = y_true_array[non_zero_mask]
        valid_y_pred = y_pred_array[non_zero_mask]
        
        percentage_errors = np.abs((valid_y_true - valid_y_pred) / valid_y_true) * 100
        mape = np.mean(percentage_errors)
        
        return {
            'mape': float(mape),
            'n': len(y_true),
            'zero_count': int(np.sum(~non_zero_mask)),
            'valid_count': int(np.sum(non_zero_mask)),
            'max_percentage_error': float(np.max(percentage_errors)),
            'min_percentage_error': float(np.min(percentage_errors)),
            'median_percentage_error': float(np.median(percentage_errors)),
            'interpretation': self._interpret_mape(mape)
        }
    
    def symmetric_mean_absolute_percentage_error(self, y_true: List[Union[int, float]], 
                                               y_pred: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
        
        SMAPE = (100/n) * Σ(2 * |y_true - y_pred|) / (|y_true| + |y_pred|)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing SMAPE and related statistics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate SMAPE for empty arrays")
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        numerator = 2 * np.abs(y_true_array - y_pred_array)
        denominator = np.abs(y_true_array) + np.abs(y_pred_array)
        
        # Handle zero denominators
        non_zero_mask = denominator != 0
        
        if not np.any(non_zero_mask):
            return {
                'smape': 0.0,  # Perfect prediction when both actual and predicted are zero
                'n': len(y_true),
                'zero_denominator_count': len(y_true),
                'valid_count': 0
            }
        
        valid_errors = numerator[non_zero_mask] / denominator[non_zero_mask] * 100
        smape = np.mean(valid_errors)
        
        return {
            'smape': float(smape),
            'n': len(y_true),
            'zero_denominator_count': int(np.sum(~non_zero_mask)),
            'valid_count': int(np.sum(non_zero_mask)),
            'max_smape_error': float(np.max(valid_errors)) if len(valid_errors) > 0 else 0.0,
            'min_smape_error': float(np.min(valid_errors)) if len(valid_errors) > 0 else 0.0,
            'interpretation': self._interpret_smape(smape)
        }
    
    def mean_absolute_scaled_error(self, y_true: List[Union[int, float]], 
                                 y_pred: List[Union[int, float]], 
                                 y_train: Optional[List[Union[int, float]]] = None) -> Dict[str, float]:
        """
        Calculate Mean Absolute Scaled Error (MASE).
        
        MASE = MAE / MAE_naive
        where MAE_naive is the MAE of a naive forecast (usually seasonal naive or random walk)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_train: Training data for calculating naive forecast baseline (optional)
            
        Returns:
            Dictionary containing MASE and related statistics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate MASE for empty arrays")
        
        # Calculate MAE of the model
        mae_result = self.mean_absolute_error(y_true, y_pred)
        mae_model = mae_result['mae']
        
        # Calculate MAE of naive forecast
        if y_train is not None and len(y_train) > 1:
            # Use training data to calculate naive forecast MAE (random walk)
            y_train_array = np.array(y_train)
            naive_errors = np.abs(y_train_array[1:] - y_train_array[:-1])
            mae_naive = np.mean(naive_errors)
        else:
            # Use in-sample naive forecast (lag-1)
            if len(y_true) < 2:
                raise ValueError("Need at least 2 data points for MASE calculation")
            
            y_true_array = np.array(y_true)
            naive_errors = np.abs(y_true_array[1:] - y_true_array[:-1])
            mae_naive = np.mean(naive_errors)
        
        if mae_naive == 0:
            mase = float('inf') if mae_model > 0 else 1.0
        else:
            mase = mae_model / mae_naive
        
        return {
            'mase': float(mase),
            'mae_model': float(mae_model),
            'mae_naive': float(mae_naive),
            'n': len(y_true),
            'interpretation': self._interpret_mase(mase)
        }
    
    def mean_squared_logarithmic_error(self, y_true: List[Union[int, float]], 
                                     y_pred: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate Mean Squared Logarithmic Error (MSLE).
        
        MSLE = (1/n) * Σ(log(1 + y_true) - log(1 + y_pred))²
        
        Args:
            y_true: Actual values (must be non-negative)
            y_pred: Predicted values (must be non-negative)
            
        Returns:
            Dictionary containing MSLE and related statistics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate MSLE for empty arrays")
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        # Check for negative values
        if np.any(y_true_array < 0) or np.any(y_pred_array < 0):
            raise ValueError("MSLE requires non-negative values")
        
        log_true = np.log1p(y_true_array)  # log(1 + y_true)
        log_pred = np.log1p(y_pred_array)  # log(1 + y_pred)
        
        squared_log_errors = (log_true - log_pred) ** 2
        msle = np.mean(squared_log_errors)
        
        # Calculate RMSLE
        rmsle = np.sqrt(msle)
        
        return {
            'msle': float(msle),
            'rmsle': float(rmsle),
            'sum_squared_log_errors': float(np.sum(squared_log_errors)),
            'n': len(y_true),
            'max_squared_log_error': float(np.max(squared_log_errors)),
            'min_squared_log_error': float(np.min(squared_log_errors))
        }
    
    def r_squared(self, y_true: List[Union[int, float]], 
                 y_pred: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate R-squared (Coefficient of Determination).
        
        R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y_true - y_pred)² and SS_tot = Σ(y_true - ȳ)²
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing R² and related statistics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate R² for empty arrays")
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        y_mean = np.mean(y_true_array)
        
        ss_res = np.sum((y_true_array - y_pred_array) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y_true_array - y_mean) ** 2)        # Total sum of squares
        
        if ss_tot == 0:
            r_squared = 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'r_squared': float(r_squared),
            'ss_res': float(ss_res),
            'ss_tot': float(ss_tot),
            'y_mean': float(y_mean),
            'n': len(y_true),
            'interpretation': self._interpret_r_squared(r_squared)
        }
    
    def adjusted_r_squared(self, y_true: List[Union[int, float]], 
                          y_pred: List[Union[int, float]], 
                          n_features: int) -> Dict[str, float]:
        """
        Calculate Adjusted R-squared.
        
        Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
        where n is sample size and p is number of features
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            n_features: Number of features/predictors
            
        Returns:
            Dictionary containing Adjusted R² and related statistics
        """
        r2_result = self.r_squared(y_true, y_pred)
        r_squared = r2_result['r_squared']
        n = len(y_true)
        
        if n <= n_features + 1:
            adj_r_squared = float('-inf')  # Undefined when n <= p + 1
        else:
            adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - n_features - 1))
        
        result = r2_result.copy()
        result.update({
            'adj_r_squared': float(adj_r_squared),
            'n_features': n_features,
            'degrees_of_freedom': n - n_features - 1
        })
        
        return result
    
    def mean_directional_accuracy(self, y_true: List[Union[int, float]], 
                                 y_pred: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate Mean Directional Accuracy (MDA).
        
        MDA measures the percentage of times the predicted direction of change
        matches the actual direction of change.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing MDA and related statistics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) < 2:
            raise ValueError("Need at least 2 data points for MDA calculation")
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        # Calculate direction of change
        true_direction = np.diff(y_true_array)
        pred_direction = np.diff(y_pred_array)
        
        # Check if directions match (same sign)
        correct_directions = np.sign(true_direction) == np.sign(pred_direction)
        mda = np.mean(correct_directions) * 100
        
        return {
            'mda': float(mda),
            'correct_predictions': int(np.sum(correct_directions)),
            'total_predictions': len(correct_directions),
            'n': len(y_true),
            'interpretation': self._interpret_mda(mda)
        }
    
    def theil_u_statistic(self, y_true: List[Union[int, float]], 
                         y_pred: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate Theil's U Statistic (Theil's Inequality Coefficient).
        
        U = √(MSE) / (√(mean(y_true²)) + √(mean(y_pred²)))
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing Theil's U and related statistics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate Theil's U for empty arrays")
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        mse = np.mean((y_true_array - y_pred_array) ** 2)
        rmse = np.sqrt(mse)
        
        mean_true_squared = np.mean(y_true_array ** 2)
        mean_pred_squared = np.mean(y_pred_array ** 2)
        
        denominator = np.sqrt(mean_true_squared) + np.sqrt(mean_pred_squared)
        
        if denominator == 0:
            theil_u = 0.0 if rmse == 0 else float('inf')
        else:
            theil_u = rmse / denominator
        
        return {
            'theil_u': float(theil_u),
            'rmse': float(rmse),
            'mean_true_squared': float(mean_true_squared),
            'mean_pred_squared': float(mean_pred_squared),
            'n': len(y_true),
            'interpretation': self._interpret_theil_u(theil_u)
        }
    
    def calculate_all_metrics(self, y_true: List[Union[int, float]], 
                            y_pred: List[Union[int, float]], 
                            n_features: int = 1) -> pd.DataFrame:
        """
        Calculate all available error metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            n_features: Number of features for adjusted R²
            
        Returns:
            DataFrame with all metrics
        """
        metrics = []
        
        try:
            mse_result = self.mean_squared_error(y_true, y_pred)
            metrics.append({'metric': 'MSE', 'value': mse_result['mse'], 'interpretation': 'Lower is better'})
        except Exception as e:
            metrics.append({'metric': 'MSE', 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        try:
            rmse_result = self.root_mean_squared_error(y_true, y_pred)
            metrics.append({'metric': 'RMSE', 'value': rmse_result['rmse'], 'interpretation': 'Lower is better'})
        except Exception as e:
            metrics.append({'metric': 'RMSE', 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        try:
            mae_result = self.mean_absolute_error(y_true, y_pred)
            metrics.append({'metric': 'MAE', 'value': mae_result['mae'], 'interpretation': 'Lower is better'})
        except Exception as e:
            metrics.append({'metric': 'MAE', 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        try:
            mape_result = self.mean_absolute_percentage_error(y_true, y_pred)
            metrics.append({'metric': 'MAPE (%)', 'value': mape_result['mape'], 'interpretation': mape_result['interpretation']})
        except Exception as e:
            metrics.append({'metric': 'MAPE (%)', 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        try:
            smape_result = self.symmetric_mean_absolute_percentage_error(y_true, y_pred)
            metrics.append({'metric': 'SMAPE (%)', 'value': smape_result['smape'], 'interpretation': smape_result['interpretation']})
        except Exception as e:
            metrics.append({'metric': 'SMAPE (%)', 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        try:
            r2_result = self.r_squared(y_true, y_pred)
            metrics.append({'metric': 'R²', 'value': r2_result['r_squared'], 'interpretation': r2_result['interpretation']})
        except Exception as e:
            metrics.append({'metric': 'R²', 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        try:
            adj_r2_result = self.adjusted_r_squared(y_true, y_pred, n_features)
            metrics.append({'metric': 'Adj R²', 'value': adj_r2_result['adj_r_squared'], 'interpretation': 'Higher is better'})
        except Exception as e:
            metrics.append({'metric': 'Adj R²', 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        # Only calculate MSLE if all values are non-negative
        if all(val >= 0 for val in y_true) and all(val >= 0 for val in y_pred):
            try:
                msle_result = self.mean_squared_logarithmic_error(y_true, y_pred)
                metrics.append({'metric': 'MSLE', 'value': msle_result['msle'], 'interpretation': 'Lower is better'})
                metrics.append({'metric': 'RMSLE', 'value': msle_result['rmsle'], 'interpretation': 'Lower is better'})
            except Exception as e:
                metrics.append({'metric': 'MSLE', 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        try:
            mda_result = self.mean_directional_accuracy(y_true, y_pred)
            metrics.append({'metric': 'MDA (%)', 'value': mda_result['mda'], 'interpretation': mda_result['interpretation']})
        except Exception as e:
            metrics.append({'metric': 'MDA (%)', 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        try:
            theil_result = self.theil_u_statistic(y_true, y_pred)
            metrics.append({'metric': "Theil's U", 'value': theil_result['theil_u'], 'interpretation': theil_result['interpretation']})
        except Exception as e:
            metrics.append({'metric': "Theil's U", 'value': f'Error: {str(e)}', 'interpretation': 'N/A'})
        
        return pd.DataFrame(metrics)
    
    def _interpret_mape(self, mape: float) -> str:
        """Interpret MAPE value."""
        if mape < 10:
            return "Excellent accuracy"
        elif mape < 20:
            return "Good accuracy"
        elif mape < 50:
            return "Reasonable accuracy"
        else:
            return "Poor accuracy"
    
    def _interpret_smape(self, smape: float) -> str:
        """Interpret SMAPE value."""
        if smape < 10:
            return "Excellent accuracy"
        elif smape < 20:
            return "Good accuracy"
        elif smape < 30:
            return "Reasonable accuracy"
        else:
            return "Poor accuracy"
    
    def _interpret_mase(self, mase: float) -> str:
        """Interpret MASE value."""
        if mase < 0.5:
            return "Excellent - much better than naive"
        elif mase < 1.0:
            return "Good - better than naive"
        elif mase == 1.0:
            return "Same as naive forecast"
        elif mase < 2.0:
            return "Poor - worse than naive"
        else:
            return "Very poor - much worse than naive"
    
    def _interpret_r_squared(self, r_squared: float) -> str:
        """Interpret R-squared value."""
        if r_squared >= 0.9:
            return "Excellent fit"
        elif r_squared >= 0.7:
            return "Good fit"
        elif r_squared >= 0.5:
            return "Moderate fit"
        elif r_squared >= 0.3:
            return "Poor fit"
        else:
            return "Very poor fit"
    
    def _interpret_mda(self, mda: float) -> str:
        """Interpret MDA value."""
        if mda >= 70:
            return "Excellent directional accuracy"
        elif mda >= 60:
            return "Good directional accuracy"
        elif mda >= 50:
            return "Fair directional accuracy"
        else:
            return "Poor directional accuracy"
    
    def _interpret_theil_u(self, theil_u: float) -> str:
        """Interpret Theil's U statistic."""
        if theil_u < 0.3:
            return "Excellent forecast"
        elif theil_u < 0.6:
            return "Good forecast"
        elif theil_u < 1.0:
            return "Reasonable forecast"
        else:
            return "Poor forecast"


def hackerrank_error_metrics_problems():
    """
    Collection of HackerRank-style problems for error metrics.
    """
    
    def problem_1_mse_calculation():
        """
        Problem 1: Calculate Mean Squared Error
        
        Given actual and predicted values, calculate MSE.
        Round to 3 decimal places.
        
        Input: y_true = [1, 2, 3, 4, 5], y_pred = [1.1, 2.2, 2.8, 4.1, 4.9]
        Expected Output: 0.020
        """
        def calculate_mse(y_true: List[Union[int, float]], 
                         y_pred: List[Union[int, float]]) -> float:
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            
            squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
            mse = sum(squared_errors) / len(squared_errors)
            return round(mse, 3)
        
        # Test case
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.2, 2.8, 4.1, 4.9]
        result = calculate_mse(y_true, y_pred)
        
        print(f"Problem 1 - Actual: {y_true}")
        print(f"Problem 1 - Predicted: {y_pred}")
        print(f"Problem 1 - MSE: {result}")
        return result
    
    def problem_2_rmse_calculation():
        """
        Problem 2: Calculate Root Mean Squared Error
        
        Given actual and predicted values, calculate RMSE.
        
        Input: y_true = [10, 20, 30, 40, 50], y_pred = [12, 18, 32, 38, 52]
        """
        def calculate_rmse(y_true: List[Union[int, float]], 
                          y_pred: List[Union[int, float]]) -> float:
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            
            mse = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)
            rmse = mse ** 0.5
            return round(rmse, 3)
        
        # Test case
        y_true = [10, 20, 30, 40, 50]
        y_pred = [12, 18, 32, 38, 52]
        result = calculate_rmse(y_true, y_pred)
        
        print(f"Problem 2 - Actual: {y_true}")
        print(f"Problem 2 - Predicted: {y_pred}")
        print(f"Problem 2 - RMSE: {result}")
        return result
    
    def problem_3_mae_calculation():
        """
        Problem 3: Calculate Mean Absolute Error
        
        Given actual and predicted values, calculate MAE.
        
        Input: y_true = [1, 3, 5, 7, 9], y_pred = [2, 3, 4, 8, 10]
        """
        def calculate_mae(y_true: List[Union[int, float]], 
                         y_pred: List[Union[int, float]]) -> float:
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            
            absolute_errors = [abs(true - pred) for true, pred in zip(y_true, y_pred)]
            mae = sum(absolute_errors) / len(absolute_errors)
            return round(mae, 3)
        
        # Test case
        y_true = [1, 3, 5, 7, 9]
        y_pred = [2, 3, 4, 8, 10]
        result = calculate_mae(y_true, y_pred)
        
        print(f"Problem 3 - Actual: {y_true}")
        print(f"Problem 3 - Predicted: {y_pred}")
        print(f"Problem 3 - MAE: {result}")
        return result
    
    def problem_4_mape_calculation():
        """
        Problem 4: Calculate Mean Absolute Percentage Error
        
        Given actual and predicted values, calculate MAPE.
        Handle zero values appropriately.
        
        Input: y_true = [100, 200, 300, 400, 500], y_pred = [110, 190, 320, 380, 520]
        """
        def calculate_mape(y_true: List[Union[int, float]], 
                          y_pred: List[Union[int, float]]) -> float:
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            
            percentage_errors = []
            for true, pred in zip(y_true, y_pred):
                if true != 0:
                    percentage_errors.append(abs((true - pred) / true) * 100)
            
            if not percentage_errors:
                return float('inf')
            
            mape = sum(percentage_errors) / len(percentage_errors)
            return round(mape, 2)
        
        # Test case
        y_true = [100, 200, 300, 400, 500]
        y_pred = [110, 190, 320, 380, 520]
        result = calculate_mape(y_true, y_pred)
        
        print(f"Problem 4 - Actual: {y_true}")
        print(f"Problem 4 - Predicted: {y_pred}")
        print(f"Problem 4 - MAPE: {result}%")
        return result
    
    def problem_5_r_squared_calculation():
        """
        Problem 5: Calculate R-squared
        
        Given actual and predicted values, calculate R².
        
        Input: y_true = [2, 4, 6, 8, 10], y_pred = [2.1, 3.9, 6.2, 7.8, 9.9]
        """
        def calculate_r_squared(y_true: List[Union[int, float]], 
                               y_pred: List[Union[int, float]]) -> float:
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 0.0
            
            y_mean = sum(y_true) / len(y_true)
            
            ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
            ss_tot = sum((true - y_mean) ** 2 for true in y_true)
            
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return round(r_squared, 4)
        
        # Test case
        y_true = [2, 4, 6, 8, 10]
        y_pred = [2.1, 3.9, 6.2, 7.8, 9.9]
        result = calculate_r_squared(y_true, y_pred)
        
        print(f"Problem 5 - Actual: {y_true}")
        print(f"Problem 5 - Predicted: {y_pred}")
        print(f"Problem 5 - R²: {result}")
        return result
    
    def problem_6_error_metric_comparison():
        """
        Problem 6: Compare Error Metrics
        
        Given two models' predictions, determine which is better based on RMSE.
        Return 1 if model 1 is better, 2 if model 2 is better, 0 if equal.
        
        Input: 
        y_true = [1, 2, 3, 4, 5]
        model1_pred = [1.1, 2.1, 2.9, 4.1, 4.9]
        model2_pred = [0.9, 2.2, 3.1, 3.8, 5.2]
        """
        def compare_models_rmse(y_true: List[Union[int, float]], 
                               model1_pred: List[Union[int, float]], 
                               model2_pred: List[Union[int, float]]) -> int:
            def calculate_rmse(y_true, y_pred):
                if len(y_true) != len(y_pred) or len(y_true) == 0:
                    return float('inf')
                mse = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)
                return mse ** 0.5
            
            rmse1 = calculate_rmse(y_true, model1_pred)
            rmse2 = calculate_rmse(y_true, model2_pred)
            
            if rmse1 < rmse2:
                return 1
            elif rmse2 < rmse1:
                return 2
            else:
                return 0
        
        # Test case
        y_true = [1, 2, 3, 4, 5]
        model1_pred = [1.1, 2.1, 2.9, 4.1, 4.9]
        model2_pred = [0.9, 2.2, 3.1, 3.8, 5.2]
        result = compare_models_rmse(y_true, model1_pred, model2_pred)
        
        # Calculate RMSEs for display
        rmse1 = (sum((t - p) ** 2 for t, p in zip(y_true, model1_pred)) / len(y_true)) ** 0.5
        rmse2 = (sum((t - p) ** 2 for t, p in zip(y_true, model2_pred)) / len(y_true)) ** 0.5
        
        print(f"Problem 6 - Actual: {y_true}")
        print(f"Problem 6 - Model 1 Pred: {model1_pred} (RMSE: {rmse1:.3f})")
        print(f"Problem 6 - Model 2 Pred: {model2_pred} (RMSE: {rmse2:.3f})")
        print(f"Problem 6 - Better Model: {result if result != 0 else 'Equal'}")
        return result
    
    # Run all problems
    print("=== HackerRank Style Error Metrics Problems ===\n")
    problem_1_mse_calculation()
    print()
    problem_2_rmse_calculation()
    print()
    problem_3_mae_calculation()
    print()
    problem_4_mape_calculation()
    print()
    problem_5_r_squared_calculation()
    print()
    problem_6_error_metric_comparison()


if __name__ == "__main__":
    # Demonstrate the error metrics calculator
    print("=== Error Metrics Calculator Demonstration ===\n")
    
    # Create sample data with different error patterns
    np.random.seed(42)
    
    # Perfect predictions
    y_true_perfect = [1, 2, 3, 4, 5]
    y_pred_perfect = [1, 2, 3, 4, 5]
    
    # Good predictions with small errors
    y_true_good = [10, 20, 30, 40, 50]
    y_pred_good = [10.5, 19.8, 30.2, 39.7, 50.3]
    
    # Poor predictions with large errors
    y_true_poor = [100, 200, 300, 400, 500]
    y_pred_poor = [120, 180, 350, 380, 520]
    
    datasets = [
        ("Perfect Predictions", y_true_perfect, y_pred_perfect),
        ("Good Predictions", y_true_good, y_pred_good),
        ("Poor Predictions", y_true_poor, y_pred_poor)
    ]
    
    print("Sample datasets:")
    for name, y_true, y_pred in datasets:
        print(f"{name}: True={y_true}, Pred={y_pred}")
    print()
    
    # Initialize calculator
    calc = ErrorMetricsCalculator()
    
    # Calculate metrics for each dataset
    for name, y_true, y_pred in datasets:
        print(f"=== {name} ===")
        
        # Calculate all metrics
        all_metrics = calc.calculate_all_metrics(y_true, y_pred, n_features=1)
        print(all_metrics.to_string(index=False))
        print()
    
    print("="*60 + "\n")
    
    # Run HackerRank-style problems
    hackerrank_error_metrics_problems()
