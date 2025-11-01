"""
Bivariate Analysis Visualization for Data Science Interview Preparation

This module provides visualization tools for understanding bivariate relationships.
Useful for explaining concepts during interviews and understanding data patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Tuple, Dict
from bivariate import BivariateAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Ellipse
    MATPLOTLIB_AVAILABLE = True
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization features will not work.")


class BivariateVisualizer:
    """
    Visualization toolkit for bivariate analysis.
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
        self.analyzer = BivariateAnalyzer()
    
    def plot_correlation_analysis(self, x: List[Union[int, float]], 
                                y: List[Union[int, float]], 
                                title: str = "Correlation Analysis",
                                save_path: Optional[str] = None):
        """
        Create a comprehensive correlation analysis plot.
        
        Args:
            x: First variable values
            y: Second variable values
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Calculate correlations
        pearson = self.analyzer.calculate_pearson_correlation(x, y)
        spearman = self.analyzer.calculate_spearman_correlation(x, y)
        
        # Plot 1: Scatter plot with regression line
        axes[0, 0].scatter(x, y, alpha=0.7, color='blue', s=50)
        
        # Add regression line
        regression = self.analyzer.simple_linear_regression(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = regression['slope'] * x_line + regression['intercept']
        axes[0, 0].plot(x_line, y_line, 'r--', linewidth=2, 
                       label=f"y = {regression['slope']:.3f}x + {regression['intercept']:.3f}")
        
        axes[0, 0].set_title(f'Scatter Plot with Regression Line\nR² = {regression["r_squared"]:.3f}')
        axes[0, 0].set_xlabel('X Variable')
        axes[0, 0].set_ylabel('Y Variable')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals plot
        residuals = regression['residuals']
        predictions = regression['predictions']
        
        axes[0, 1].scatter(predictions, residuals, alpha=0.7, color='green', s=50)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Correlation comparison
        correlations = ['Pearson', 'Spearman']
        values = [pearson['correlation'], spearman['correlation']]
        colors = ['skyblue', 'lightgreen']
        
        bars = axes[1, 0].bar(correlations, values, color=colors, alpha=0.7)
        axes[1, 0].set_title('Correlation Comparison')
        axes[1, 0].set_ylabel('Correlation Coefficient')
        axes[1, 0].set_ylim(-1.1, 1.1)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.05 * np.sign(height),
                           f'{value:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 4: Statistics summary
        axes[1, 1].axis('off')
        
        # Create statistics text
        stats_text = f"""
        Correlation Analysis Summary:
        
        Pearson Correlation:
        • r = {pearson['correlation']:.4f}
        • Strength: {pearson['strength']}
        • Direction: {pearson['direction']}
        • Interpretation: {pearson['interpretation']}
        
        Spearman Correlation:
        • ρ = {spearman['correlation']:.4f}
        • Strength: {spearman['strength']}
        • Direction: {spearman['direction']}
        • Interpretation: {spearman['interpretation']}
        
        Regression Analysis:
        • Slope: {regression['slope']:.4f}
        • Intercept: {regression['intercept']:.4f}
        • R-squared: {regression['r_squared']:.4f}
        • RMSE: {np.sqrt(regression['mse']):.4f}
        
        Data Summary:
        • Sample size: {len(x)}
        • X range: [{min(x):.2f}, {max(x):.2f}]
        • Y range: [{min(y):.2f}, {max(y):.2f}]
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_regression_diagnostics(self, x: List[Union[int, float]], 
                                  y: List[Union[int, float]], 
                                  title: str = "Regression Diagnostics",
                                  save_path: Optional[str] = None):
        """
        Create comprehensive regression diagnostic plots.
        
        Args:
            x: Independent variable values
            y: Dependent variable values
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Perform regression
        regression = self.analyzer.simple_linear_regression(x, y)
        residuals = np.array(regression['residuals'])
        predictions = np.array(regression['predictions'])
        
        # Plot 1: Fitted vs Residuals
        axes[0, 0].scatter(predictions, residuals, alpha=0.7, color='blue', s=50)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Fitted vs Residuals')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line to residuals
        try:
            z = np.polyfit(predictions, residuals, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(predictions, p(predictions), "g--", alpha=0.8, 
                           label=f'Trend: slope={z[0]:.4f}')
            axes[0, 0].legend()
        except:
            pass
        
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
        
        # Plot 3: Scale-Location plot
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[1, 0].scatter(predictions, sqrt_abs_residuals, alpha=0.7, color='green', s=50)
        axes[1, 0].set_title('Scale-Location Plot')
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Residuals|')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add trend line
        try:
            z = np.polyfit(predictions, sqrt_abs_residuals, 1)
            p = np.poly1d(z)
            axes[1, 0].plot(predictions, p(predictions), "r--", alpha=0.8)
        except:
            pass
        
        # Plot 4: Residuals histogram
        axes[1, 1].hist(residuals, bins=min(15, len(residuals)//2), alpha=0.7, 
                       color='purple', density=True)
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = ((1 / (sigma * np.sqrt(2 * np.pi))) * 
                 np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2))
        axes[1, 1].plot(x_norm, y_norm, 'r--', linewidth=2, label='Normal Distribution')
        
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, data: Dict[str, List[Union[int, float]]], 
                              title: str = "Correlation Matrix",
                              save_path: Optional[str] = None):
        """
        Create a correlation matrix heatmap.
        
        Args:
            data: Dictionary with variable names as keys and data as values
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Calculate correlation matrix
        corr_matrix = self.analyzer.correlation_matrix(data)
        
        # Plot 1: Heatmap
        im = ax1.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        # Set ticks and labels
        ax1.set_xticks(range(len(corr_matrix.columns)))
        ax1.set_yticks(range(len(corr_matrix.index)))
        ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax1.set_yticklabels(corr_matrix.index)
        
        # Add correlation values as text
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                value = corr_matrix.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax1.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color=color, fontweight='bold')
        
        ax1.set_title('Correlation Heatmap')
        
        # Plot 2: Correlation strength distribution
        # Get upper triangle values (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix.values, dtype=bool), k=1)
        upper_triangle_values = corr_matrix.values[mask]
        
        ax2.hist(upper_triangle_values, bins=15, alpha=0.7, color='skyblue', 
                edgecolor='black', density=True)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Correlation')
        ax2.axvline(x=np.mean(upper_triangle_values), color='green', linestyle='--', 
                   alpha=0.7, label=f'Mean: {np.mean(upper_triangle_values):.3f}')
        
        ax2.set_title('Distribution of Correlations')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_relationship_types(self, save_path: Optional[str] = None):
        """
        Demonstrate different types of bivariate relationships.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Types of Bivariate Relationships', fontsize=16, fontweight='bold')
        
        np.random.seed(42)
        n_points = 50
        
        # 1. Perfect Linear Positive
        x1 = np.linspace(1, 10, n_points)
        y1 = 2 * x1 + 1
        self._plot_relationship(axes[0, 0], x1, y1, "Perfect Linear Positive\nr = 1.00")
        
        # 2. Strong Linear Positive with noise
        x2 = np.linspace(1, 10, n_points)
        y2 = 2 * x2 + 1 + np.random.normal(0, 2, n_points)
        corr2 = self.analyzer.calculate_pearson_correlation(x2.tolist(), y2.tolist())
        self._plot_relationship(axes[0, 1], x2, y2, f"Strong Linear Positive\nr = {corr2['correlation']:.2f}")
        
        # 3. Perfect Linear Negative
        x3 = np.linspace(1, 10, n_points)
        y3 = -2 * x3 + 20
        self._plot_relationship(axes[0, 2], x3, y3, "Perfect Linear Negative\nr = -1.00")
        
        # 4. Non-linear (Quadratic)
        x4 = np.linspace(-5, 5, n_points)
        y4 = x4 ** 2
        corr4 = self.analyzer.calculate_pearson_correlation(x4.tolist(), y4.tolist())
        spearman4 = self.analyzer.calculate_spearman_correlation(x4.tolist(), y4.tolist())
        self._plot_relationship(axes[1, 0], x4, y4, 
                              f"Non-linear (Quadratic)\nPearson r = {corr4['correlation']:.2f}\nSpearman ρ = {spearman4['correlation']:.2f}")
        
        # 5. No Correlation
        x5 = np.random.normal(5, 2, n_points)
        y5 = np.random.normal(10, 3, n_points)
        corr5 = self.analyzer.calculate_pearson_correlation(x5.tolist(), y5.tolist())
        self._plot_relationship(axes[1, 1], x5, y5, f"No Correlation\nr = {corr5['correlation']:.2f}")
        
        # 6. Outlier Effect
        x6 = np.linspace(1, 10, n_points-1)
        y6 = 2 * x6 + 1 + np.random.normal(0, 0.5, n_points-1)
        # Add outlier
        x6 = np.append(x6, 15)
        y6 = np.append(y6, 5)
        corr6 = self.analyzer.calculate_pearson_correlation(x6.tolist(), y6.tolist())
        
        axes[1, 2].scatter(x6[:-1], y6[:-1], alpha=0.7, color='blue', s=50, label='Normal points')
        axes[1, 2].scatter(x6[-1], y6[-1], color='red', s=100, marker='^', label='Outlier')
        
        # Add regression line
        reg6 = self.analyzer.simple_linear_regression(x6.tolist(), y6.tolist())
        x_line = np.linspace(min(x6), max(x6), 100)
        y_line = reg6['slope'] * x_line + reg6['intercept']
        axes[1, 2].plot(x_line, y_line, 'r--', alpha=0.7)
        
        axes[1, 2].set_title(f"Effect of Outliers\nr = {corr6['correlation']:.2f}")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def _plot_relationship(self, ax, x, y, title):
        """Helper method to plot a single relationship."""
        ax.scatter(x, y, alpha=0.7, color='blue', s=50)
        
        # Add regression line
        try:
            reg = self.analyzer.simple_linear_regression(x.tolist(), y.tolist())
            x_line = np.linspace(min(x), max(x), 100)
            y_line = reg['slope'] * x_line + reg['intercept']
            ax.plot(x_line, y_line, 'r--', alpha=0.7)
        except:
            pass
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    def plot_goodness_of_fit_comparison(self, datasets: List[Tuple[str, List, List]], 
                                      save_path: Optional[str] = None):
        """
        Compare goodness of fit across multiple datasets.
        
        Args:
            datasets: List of tuples (name, x_data, y_data)
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Goodness of Fit Comparison', fontsize=16, fontweight='bold')
        
        # Calculate metrics for all datasets
        metrics = []
        for name, x, y in datasets:
            reg = self.analyzer.simple_linear_regression(x, y)
            gof = self.analyzer.calculate_goodness_of_fit(y, reg['predictions'])
            metrics.append({
                'name': name,
                'r_squared': gof['r_squared'],
                'rmse': gof['rmse'],
                'mae': gof['mae'],
                'mape': gof['mape']
            })
        
        # Plot 1: R-squared comparison
        names = [m['name'] for m in metrics]
        r_squared_values = [m['r_squared'] for m in metrics]
        
        bars1 = axes[0, 0].bar(names, r_squared_values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('R-squared Comparison')
        axes[0, 0].set_ylabel('R-squared')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, r_squared_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: RMSE comparison
        rmse_values = [m['rmse'] for m in metrics]
        bars2 = axes[0, 1].bar(names, rmse_values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, rmse_values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(rmse_values)*0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 3: MAE comparison
        mae_values = [m['mae'] for m in metrics]
        bars3 = axes[1, 0].bar(names, mae_values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, mae_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(mae_values)*0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 4: Summary table
        axes[1, 1].axis('off')
        
        table_data = []
        for m in metrics:
            table_data.append([
                m['name'],
                f"{m['r_squared']:.3f}",
                f"{m['rmse']:.2f}",
                f"{m['mae']:.2f}",
                f"{m['mape']:.1f}%"
            ])
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Dataset', 'R²', 'RMSE', 'MAE', 'MAPE'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Goodness of Fit Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


def demonstrate_bivariate_visualization():
    """Demonstrate bivariate visualization capabilities."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot run visualization demonstration.")
        return
    
    print("=== Bivariate Analysis Visualization Demonstration ===\n")
    
    # Create sample datasets
    np.random.seed(42)
    
    # Dataset 1: Strong linear relationship
    x1 = np.linspace(1, 10, 50)
    y1 = 2 * x1 + 1 + np.random.normal(0, 1, 50)
    
    # Dataset 2: Non-linear relationship
    x2 = np.linspace(-3, 3, 50)
    y2 = x2 ** 2 + np.random.normal(0, 0.5, 50)
    
    # Dataset 3: Weak correlation
    x3 = np.random.normal(5, 2, 50)
    y3 = 0.3 * x3 + np.random.normal(10, 3, 50)
    
    print("Created sample datasets for demonstration")
    print("Dataset 1: Strong linear relationship")
    print("Dataset 2: Non-linear (quadratic) relationship")
    print("Dataset 3: Weak linear relationship")
    print()
    
    # Initialize visualizer
    visualizer = BivariateVisualizer()
    
    print("Creating visualization plots...")
    
    print("1. Correlation analysis for strong linear relationship")
    visualizer.plot_correlation_analysis(x1.tolist(), y1.tolist(), 
                                       "Strong Linear Relationship Analysis")
    
    print("2. Regression diagnostics")
    visualizer.plot_regression_diagnostics(x1.tolist(), y1.tolist(),
                                         "Regression Diagnostics - Linear Data")
    
    print("3. Different relationship types demonstration")
    visualizer.plot_relationship_types()
    
    print("4. Correlation matrix for multiple variables")
    multi_data = {
        'X1': x1.tolist(),
        'Y1': y1.tolist(),
        'X2': x2.tolist(),
        'Y2': y2.tolist()
    }
    visualizer.plot_correlation_matrix(multi_data, "Multi-variable Correlation Matrix")
    
    print("5. Goodness of fit comparison")
    datasets = [
        ("Strong Linear", x1.tolist(), y1.tolist()),
        ("Non-linear", x2.tolist(), y2.tolist()),
        ("Weak Linear", x3.tolist(), y3.tolist())
    ]
    visualizer.plot_goodness_of_fit_comparison(datasets)
    
    print("Bivariate visualization demonstration completed!")


if __name__ == "__main__":
    demonstrate_bivariate_visualization()
