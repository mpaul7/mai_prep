"""
Error Metrics Visualization for Data Science Interview Preparation

This module provides visualization tools for understanding error metrics.
Useful for explaining concepts during interviews and understanding model performance.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Tuple, Dict
from error_metrics import ErrorMetricsCalculator
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization features will not work.")


class ErrorMetricsVisualizer:
    """
    Visualization toolkit for error metrics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib and seaborn are required for visualization. Install with: pip install matplotlib seaborn")
        
        self.figsize = figsize
        self.calc = ErrorMetricsCalculator()
    
    def plot_error_analysis(self, y_true: List[Union[int, float]], 
                           y_pred: List[Union[int, float]], 
                           title: str = "Error Analysis",
                           save_path: Optional[str] = None):
        """
        Create a comprehensive error analysis plot.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        errors = y_true_array - y_pred_array
        abs_errors = np.abs(errors)
        
        # Calculate metrics
        mse_result = self.calc.mean_squared_error(y_true, y_pred)
        mae_result = self.calc.mean_absolute_error(y_true, y_pred)
        r2_result = self.calc.r_squared(y_true, y_pred)
        
        # Plot 1: Actual vs Predicted
        axes[0, 0].scatter(y_true_array, y_pred_array, alpha=0.7, color='blue', s=50)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', 
                       linewidth=2, label='Perfect Prediction')
        
        axes[0, 0].set_title(f'Actual vs Predicted\nR² = {r2_result["r_squared"]:.3f}')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals plot
        axes[0, 1].scatter(y_pred_array, errors, alpha=0.7, color='green', s=50)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals (Actual - Predicted)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        axes[1, 0].hist(errors, bins=min(15, len(errors)//2), alpha=0.7, 
                       color='purple', edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        axes[1, 0].axvline(x=np.mean(errors), color='green', linestyle='--', 
                          alpha=0.7, label=f'Mean Error: {np.mean(errors):.3f}')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].set_xlabel('Error (Actual - Predicted)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Error metrics summary
        axes[1, 1].axis('off')
        
        # Calculate additional metrics
        try:
            mape_result = self.calc.mean_absolute_percentage_error(y_true, y_pred)
            mape_text = f"{mape_result['mape']:.2f}%"
        except:
            mape_text = "N/A"
        
        try:
            rmse_result = self.calc.root_mean_squared_error(y_true, y_pred)
            rmse_text = f"{rmse_result['rmse']:.4f}"
        except:
            rmse_text = "N/A"
        
        metrics_text = f"""
        Error Metrics Summary:
        
        Mean Squared Error (MSE):
        • Value: {mse_result['mse']:.4f}
        • Interpretation: Lower is better
        
        Root Mean Squared Error (RMSE):
        • Value: {rmse_text}
        • Interpretation: Same units as target
        
        Mean Absolute Error (MAE):
        • Value: {mae_result['mae']:.4f}
        • Interpretation: Average absolute error
        
        Mean Absolute Percentage Error (MAPE):
        • Value: {mape_text}
        • Interpretation: Percentage-based error
        
        R-squared (R²):
        • Value: {r2_result['r_squared']:.4f}
        • Interpretation: {r2_result['interpretation']}
        
        Data Summary:
        • Sample size: {len(y_true)}
        • Mean error: {np.mean(errors):.4f}
        • Std error: {np.std(errors):.4f}
        • Max absolute error: {np.max(abs_errors):.4f}
        """
        
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, models: List[Tuple[str, List, List]], 
                              title: str = "Model Comparison",
                              save_path: Optional[str] = None):
        """
        Compare error metrics across multiple models.
        
        Args:
            models: List of tuples (model_name, y_true, y_pred)
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Calculate metrics for all models
        metrics_data = []
        for name, y_true, y_pred in models:
            mse = self.calc.mean_squared_error(y_true, y_pred)
            mae = self.calc.mean_absolute_error(y_true, y_pred)
            r2 = self.calc.r_squared(y_true, y_pred)
            
            try:
                mape = self.calc.mean_absolute_percentage_error(y_true, y_pred)
                mape_value = mape['mape']
            except:
                mape_value = float('inf')
            
            metrics_data.append({
                'model': name,
                'mse': mse['mse'],
                'mae': mae['mae'],
                'r2': r2['r_squared'],
                'mape': mape_value
            })
        
        model_names = [m['model'] for m in metrics_data]
        
        # Plot 1: MSE comparison
        mse_values = [m['mse'] for m in metrics_data]
        bars1 = axes[0, 0].bar(model_names, mse_values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Mean Squared Error (MSE)')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, mse_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(mse_values)*0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 2: MAE comparison
        mae_values = [m['mae'] for m in metrics_data]
        bars2 = axes[0, 1].bar(model_names, mae_values, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Mean Absolute Error (MAE)')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, mae_values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(mae_values)*0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 3: R² comparison
        r2_values = [m['r2'] for m in metrics_data]
        bars3 = axes[1, 0].bar(model_names, r2_values, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('R-squared (R²)')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_ylim(min(0, min(r2_values) - 0.1), 1.1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, r2_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 4: Summary table
        axes[1, 1].axis('off')
        
        table_data = []
        for m in metrics_data:
            mape_str = f"{m['mape']:.2f}%" if m['mape'] != float('inf') else "N/A"
            table_data.append([
                m['model'],
                f"{m['mse']:.4f}",
                f"{m['mae']:.4f}",
                f"{m['r2']:.3f}",
                mape_str
            ])
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Model', 'MSE', 'MAE', 'R²', 'MAPE'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Metrics Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_error_sensitivity(self, y_true: List[Union[int, float]], 
                             save_path: Optional[str] = None):
        """
        Demonstrate sensitivity of different error metrics to outliers.
        
        Args:
            y_true: Actual values
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Error Metrics Sensitivity to Outliers', fontsize=16, fontweight='bold')
        
        y_true_array = np.array(y_true)
        
        # Create predictions with increasing outlier magnitude
        outlier_magnitudes = np.linspace(0, 5, 50)
        mse_values = []
        mae_values = []
        
        for magnitude in outlier_magnitudes:
            # Create predictions with one outlier
            y_pred = y_true_array.copy().astype(float)
            if len(y_pred) > 0:
                y_pred[-1] += magnitude  # Add outlier to last prediction
            
            mse = self.calc.mean_squared_error(y_true, y_pred.tolist())
            mae = self.calc.mean_absolute_error(y_true, y_pred.tolist())
            
            mse_values.append(mse['mse'])
            mae_values.append(mae['mae'])
        
        # Plot 1: MSE sensitivity
        axes[0, 0].plot(outlier_magnitudes, mse_values, 'b-', linewidth=2, label='MSE')
        axes[0, 0].set_title('MSE Sensitivity to Outliers')
        axes[0, 0].set_xlabel('Outlier Magnitude')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: MAE sensitivity
        axes[0, 1].plot(outlier_magnitudes, mae_values, 'g-', linewidth=2, label='MAE')
        axes[0, 1].set_title('MAE Sensitivity to Outliers')
        axes[0, 1].set_xlabel('Outlier Magnitude')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: MSE vs MAE comparison
        axes[1, 0].plot(outlier_magnitudes, mse_values, 'b-', linewidth=2, label='MSE')
        axes[1, 0].plot(outlier_magnitudes, mae_values, 'g-', linewidth=2, label='MAE')
        axes[1, 0].set_title('MSE vs MAE Sensitivity Comparison')
        axes[1, 0].set_xlabel('Outlier Magnitude')
        axes[1, 0].set_ylabel('Error Value')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Ratio of MSE to MAE
        mse_mae_ratio = [mse/mae if mae != 0 else 0 for mse, mae in zip(mse_values, mae_values)]
        axes[1, 1].plot(outlier_magnitudes, mse_mae_ratio, 'r-', linewidth=2)
        axes[1, 1].set_title('MSE/MAE Ratio')
        axes[1, 1].set_xlabel('Outlier Magnitude')
        axes[1, 1].set_ylabel('MSE/MAE Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add explanation text
        explanation = """
        Key Insights:
        • MSE increases quadratically with outlier magnitude
        • MAE increases linearly with outlier magnitude
        • MSE/MAE ratio shows MSE's higher sensitivity
        • Use MAE when outliers are expected
        • Use MSE when large errors are particularly bad
        """
        
        axes[1, 1].text(0.02, 0.98, explanation, transform=axes[1, 1].transAxes, 
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_percentage_metrics_comparison(self, datasets: List[Tuple[str, List, List]], 
                                         save_path: Optional[str] = None):
        """
        Compare percentage-based error metrics across different scales.
        
        Args:
            datasets: List of tuples (name, y_true, y_pred)
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Percentage-based Error Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Calculate metrics for all datasets
        results = []
        for name, y_true, y_pred in datasets:
            mae = self.calc.mean_absolute_error(y_true, y_pred)
            
            try:
                mape = self.calc.mean_absolute_percentage_error(y_true, y_pred)
                mape_value = mape['mape']
            except:
                mape_value = float('inf')
            
            try:
                smape = self.calc.symmetric_mean_absolute_percentage_error(y_true, y_pred)
                smape_value = smape['smape']
            except:
                smape_value = float('inf')
            
            results.append({
                'name': name,
                'mae': mae['mae'],
                'mape': mape_value,
                'smape': smape_value,
                'scale': np.mean(y_true)
            })
        
        names = [r['name'] for r in results]
        
        # Plot 1: MAE comparison
        mae_values = [r['mae'] for r in results]
        axes[0, 0].bar(names, mae_values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: MAPE comparison
        mape_values = [r['mape'] if r['mape'] != float('inf') else 0 for r in results]
        axes[0, 1].bar(names, mape_values, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Mean Absolute Percentage Error (MAPE)')
        axes[0, 1].set_ylabel('MAPE (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: SMAPE comparison
        smape_values = [r['smape'] if r['smape'] != float('inf') else 0 for r in results]
        axes[1, 0].bar(names, smape_values, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Symmetric Mean Absolute Percentage Error (SMAPE)')
        axes[1, 0].set_ylabel('SMAPE (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Scale vs Error relationship
        scales = [r['scale'] for r in results]
        axes[1, 1].scatter(scales, mae_values, s=100, alpha=0.7, label='MAE')
        
        # Add labels for each point
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (scales[i], mae_values[i]), 
                               xytext=(5, 5), textcoords='offset points')
        
        axes[1, 1].set_title('Scale vs MAE Relationship')
        axes[1, 1].set_xlabel('Data Scale (Mean of True Values)')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_residual_analysis(self, y_true: List[Union[int, float]], 
                             y_pred: List[Union[int, float]], 
                             title: str = "Residual Analysis",
                             save_path: Optional[str] = None):
        """
        Create detailed residual analysis plots.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        residuals = y_true_array - y_pred_array
        
        # Plot 1: Residuals vs Fitted
        axes[0, 0].scatter(y_pred_array, residuals, alpha=0.7, color='blue', s=50)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Residuals vs Fitted Values')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot of residuals
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot of Residuals')
            axes[0, 1].grid(True, alpha=0.3)
        except ImportError:
            # Manual Q-Q plot
            sorted_residuals = np.sort(residuals)
            n = len(residuals)
            theoretical_quantiles = [stats.norm.ppf((i + 0.5) / n) for i in range(n)]
            
            axes[0, 1].scatter(theoretical_quantiles, sorted_residuals, alpha=0.7, color='blue')
            axes[0, 1].plot(theoretical_quantiles, theoretical_quantiles, 'r--', alpha=0.7)
            axes[0, 1].set_title('Q-Q Plot of Residuals')
            axes[0, 1].set_xlabel('Theoretical Quantiles')
            axes[0, 1].set_ylabel('Sample Quantiles')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Histogram of residuals
        axes[1, 0].hist(residuals, bins=min(15, len(residuals)//2), alpha=0.7, 
                       color='green', edgecolor='black', density=True)
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = ((1 / (sigma * np.sqrt(2 * np.pi))) * 
             np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        axes[1, 0].plot(x, y, 'r--', linewidth=2, label='Normal Distribution')
        
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residual statistics
        axes[1, 1].axis('off')
        
        # Calculate residual statistics
        residual_stats = {
            'Mean': np.mean(residuals),
            'Std Dev': np.std(residuals),
            'Min': np.min(residuals),
            'Max': np.max(residuals),
            'Q1': np.percentile(residuals, 25),
            'Median': np.median(residuals),
            'Q3': np.percentile(residuals, 75),
            'Skewness': stats.skew(residuals) if 'stats' in globals() else 'N/A',
            'Kurtosis': stats.kurtosis(residuals) if 'stats' in globals() else 'N/A'
        }
        
        stats_text = "Residual Statistics:\n\n"
        for stat, value in residual_stats.items():
            if isinstance(value, (int, float)):
                stats_text += f"{stat}: {value:.4f}\n"
            else:
                stats_text += f"{stat}: {value}\n"
        
        stats_text += f"\nInterpretation:\n"
        stats_text += f"• Mean ≈ 0: {'✓' if abs(residual_stats['Mean']) < 0.1 else '✗'}\n"
        stats_text += f"• Symmetric: {'✓' if abs(residual_stats['Mean']) < 0.1 else '✗'}\n"
        stats_text += f"• Homoscedastic: Check scatter plot\n"
        stats_text += f"• Normal: Check Q-Q plot & histogram"
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


def demonstrate_error_visualization():
    """Demonstrate error metrics visualization capabilities."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot run visualization demonstration.")
        return
    
    print("=== Error Metrics Visualization Demonstration ===\n")
    
    # Create sample datasets
    np.random.seed(42)
    
    # Perfect predictions
    y_true_perfect = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_pred_perfect = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Good predictions with small errors
    y_true_good = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    y_pred_good = [10.5, 19.8, 30.2, 39.7, 50.3, 59.8, 70.1, 79.9, 90.2, 99.8]
    
    # Poor predictions with large errors
    y_true_poor = [100, 200, 300, 400, 500]
    y_pred_poor = [120, 180, 350, 380, 520]
    
    # Different scales
    small_scale = ([1, 2, 3, 4, 5], [1.1, 2.1, 2.9, 4.1, 4.9])
    large_scale = ([1000, 2000, 3000, 4000, 5000], [1100, 2100, 2900, 4100, 4900])
    
    print("Created sample datasets for demonstration")
    print("Dataset 1: Perfect predictions")
    print("Dataset 2: Good predictions with small errors")
    print("Dataset 3: Poor predictions with large errors")
    print("Dataset 4: Different scales for percentage metrics")
    print()
    
    # Initialize visualizer
    visualizer = ErrorMetricsVisualizer()
    
    print("Creating visualization plots...")
    
    print("1. Error analysis for good predictions")
    visualizer.plot_error_analysis(y_true_good, y_pred_good, "Good Predictions - Error Analysis")
    
    print("2. Model comparison")
    models = [
        ("Perfect Model", y_true_perfect, y_pred_perfect),
        ("Good Model", y_true_good, y_pred_good),
        ("Poor Model", y_true_poor, y_pred_poor)
    ]
    visualizer.plot_metrics_comparison(models, "Model Performance Comparison")
    
    print("3. Error sensitivity analysis")
    visualizer.plot_error_sensitivity(y_true_good, "Error Metrics Sensitivity Analysis")
    
    print("4. Percentage metrics comparison")
    percentage_datasets = [
        ("Small Scale", small_scale[0], small_scale[1]),
        ("Large Scale", large_scale[0], large_scale[1])
    ]
    visualizer.plot_percentage_metrics_comparison(percentage_datasets)
    
    print("5. Residual analysis")
    visualizer.plot_residual_analysis(y_true_good, y_pred_good, "Residual Analysis - Good Model")
    
    print("Error metrics visualization demonstration completed!")


if __name__ == "__main__":
    demonstrate_error_visualization()
