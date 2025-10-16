"""
Outlier Visualization for Data Science Interview Preparation

This module provides visualization tools for understanding outlier detection methods.
Useful for explaining concepts during interviews and understanding data patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Tuple
from outliers import OutlierDetector
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class OutlierVisualizer:
    """
    Visualization toolkit for outlier detection methods.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.detector = OutlierDetector()
    
    def plot_outlier_detection_comparison(self, data: List[Union[int, float]], 
                                        save_path: Optional[str] = None):
        """
        Create a comprehensive comparison plot of different outlier detection methods.
        
        Args:
            data: List of numeric values
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Outlier Detection Methods Comparison', fontsize=16, fontweight='bold')
        
        # Method 1: Z-Score
        try:
            z_result = self.detector.z_score_outliers(data, threshold=2.0)
            self._plot_single_method(axes[0, 0], data, z_result['outlier_indices'], 
                                   'Z-Score Method', 'Z-score > 2.0')
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f'Z-Score Error: {str(e)}', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Z-Score Method')
        
        # Method 2: Modified Z-Score
        try:
            mod_z_result = self.detector.modified_z_score_outliers(data)
            self._plot_single_method(axes[0, 1], data, mod_z_result['outlier_indices'], 
                                   'Modified Z-Score Method', 'Modified Z-score > 3.5')
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Modified Z-Score Error: {str(e)}', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Modified Z-Score Method')
        
        # Method 3: IQR
        try:
            iqr_result = self.detector.iqr_outliers(data)
            self._plot_single_method(axes[0, 2], data, iqr_result['outlier_indices'], 
                                   'IQR Method', f'k = {iqr_result["k"]}')
        except Exception as e:
            axes[0, 2].text(0.5, 0.5, f'IQR Error: {str(e)}', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('IQR Method')
        
        # Method 4: Percentile
        try:
            perc_result = self.detector.percentile_outliers(data, 5, 95)
            self._plot_single_method(axes[1, 0], data, perc_result['outlier_indices'], 
                                   'Percentile Method', '5th-95th percentile')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Percentile Error: {str(e)}', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Percentile Method')
        
        # Method 5: Isolation Forest (if enough data)
        if len(data) >= 10:
            try:
                iso_result = self.detector.isolation_forest_outliers(data)
                self._plot_single_method(axes[1, 1], data, iso_result['outlier_indices'], 
                                       'Isolation Forest', f'contamination = {iso_result["contamination"]}')
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Isolation Forest Error: {str(e)}', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Isolation Forest')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\n(need ≥10 points)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Isolation Forest')
        
        # Method comparison summary
        try:
            comparison = self.detector.compare_methods(data)
            self._plot_method_summary(axes[1, 2], comparison)
        except Exception as e:
            axes[1, 2].text(0.5, 0.5, f'Comparison Error: {str(e)}', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Method Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def _plot_single_method(self, ax, data: List[Union[int, float]], 
                          outlier_indices: List[int], title: str, subtitle: str):
        """Helper method to plot a single outlier detection method."""
        # Create scatter plot
        normal_indices = [i for i in range(len(data)) if i not in outlier_indices]
        
        if normal_indices:
            ax.scatter(normal_indices, [data[i] for i in normal_indices], 
                      c='blue', alpha=0.6, label='Normal', s=50)
        
        if outlier_indices:
            ax.scatter(outlier_indices, [data[i] for i in outlier_indices], 
                      c='red', alpha=0.8, label='Outlier', s=80, marker='^')
        
        ax.set_title(f'{title}\n({subtitle})', fontsize=10)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_method_summary(self, ax, comparison: pd.DataFrame):
        """Helper method to plot method comparison summary."""
        methods = comparison['method'].tolist()
        counts = []
        
        for count in comparison['outliers_count']:
            if isinstance(count, (int, float)):
                counts.append(count)
            else:
                counts.append(0)  # Error cases
        
        bars = ax.bar(methods, counts, color=['skyblue', 'lightgreen', 'lightcoral', 
                                            'lightyellow', 'lightpink'][:len(methods)])
        
        ax.set_title('Outliers Detected by Method')
        ax.set_ylabel('Number of Outliers')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
    
    def plot_boxplot_with_outliers(self, data: List[Union[int, float]], 
                                  save_path: Optional[str] = None):
        """
        Create a boxplot showing outliers and IQR boundaries.
        
        Args:
            data: List of numeric values
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Standard boxplot
        box_plot = ax1.boxplot(data, patch_artist=True, notch=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][0].set_alpha(0.7)
        
        ax1.set_title('Boxplot with Outliers', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Add IQR information
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        ax1.axhline(y=lower_bound, color='red', linestyle='--', alpha=0.7, 
                   label=f'Lower bound: {lower_bound:.2f}')
        ax1.axhline(y=upper_bound, color='red', linestyle='--', alpha=0.7, 
                   label=f'Upper bound: {upper_bound:.2f}')
        ax1.legend()
        
        # Scatter plot with outlier highlighting
        iqr_result = self.detector.iqr_outliers(data)
        normal_indices = [i for i in range(len(data)) if i not in iqr_result['outlier_indices']]
        
        if normal_indices:
            ax2.scatter(normal_indices, [data[i] for i in normal_indices], 
                       c='blue', alpha=0.6, label='Normal', s=50)
        
        if iqr_result['outlier_indices']:
            ax2.scatter(iqr_result['outlier_indices'], 
                       [data[i] for i in iqr_result['outlier_indices']], 
                       c='red', alpha=0.8, label='Outlier', s=80, marker='^')
        
        ax2.axhline(y=lower_bound, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=upper_bound, color='red', linestyle='--', alpha=0.7)
        ax2.fill_between(range(len(data)), lower_bound, upper_bound, alpha=0.2, color='green')
        
        ax2.set_title('Scatter Plot with IQR Boundaries', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_distribution_with_outliers(self, data: List[Union[int, float]], 
                                      method: str = 'iqr', save_path: Optional[str] = None):
        """
        Plot data distribution with outliers highlighted.
        
        Args:
            data: List of numeric values
            method: Outlier detection method ('iqr', 'z_score', 'modified_z_score')
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Get outliers based on method
        if method == 'iqr':
            result = self.detector.iqr_outliers(data)
        elif method == 'z_score':
            result = self.detector.z_score_outliers(data)
        elif method == 'modified_z_score':
            result = self.detector.modified_z_score_outliers(data)
        else:
            raise ValueError("Method must be 'iqr', 'z_score', or 'modified_z_score'")
        
        outlier_indices = result['outlier_indices']
        normal_data = [data[i] for i in range(len(data)) if i not in outlier_indices]
        outlier_data = [data[i] for i in outlier_indices]
        
        # Histogram
        ax1.hist(normal_data, bins=20, alpha=0.7, color='blue', label='Normal Data', density=True)
        if outlier_data:
            ax1.hist(outlier_data, bins=10, alpha=0.7, color='red', label='Outliers', density=True)
        
        ax1.set_title(f'Distribution with Outliers ({method.replace("_", " ").title()})', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Highlight outliers in Q-Q plot
        if outlier_indices:
            sorted_data = sorted(data)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
            
            for idx in outlier_indices:
                value = data[idx]
                sorted_idx = sorted_data.index(value)
                if sorted_idx < len(theoretical_quantiles):
                    ax2.scatter(theoretical_quantiles[sorted_idx], value, 
                              c='red', s=100, marker='^', zorder=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_outlier_impact_analysis(self, data: List[Union[int, float]], 
                                   save_path: Optional[str] = None):
        """
        Visualize the impact of outliers on statistical measures.
        
        Args:
            data: List of numeric values
            save_path: Optional path to save the plot
        """
        # Get outliers using IQR method
        iqr_result = self.detector.iqr_outliers(data)
        impact = self.detector.analyze_outlier_impact(data, iqr_result['outlier_indices'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Impact of Outliers on Statistical Measures', fontsize=16, fontweight='bold')
        
        # Mean comparison
        means = [impact['original_mean'], impact['clean_mean']]
        labels = ['With Outliers', 'Without Outliers']
        colors = ['lightcoral', 'lightblue']
        
        bars1 = axes[0, 0].bar(labels, means, color=colors)
        axes[0, 0].set_title('Mean Comparison')
        axes[0, 0].set_ylabel('Mean Value')
        
        # Add value labels
        for bar, mean in zip(bars1, means):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{mean:.2f}', ha='center', va='bottom')
        
        # Standard deviation comparison
        stds = [impact['original_std'], impact['clean_std']]
        bars2 = axes[0, 1].bar(labels, stds, color=colors)
        axes[0, 1].set_title('Standard Deviation Comparison')
        axes[0, 1].set_ylabel('Standard Deviation')
        
        for bar, std in zip(bars2, stds):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{std:.2f}', ha='center', va='bottom')
        
        # Data distribution comparison
        clean_data = [data[i] for i in range(len(data)) if i not in iqr_result['outlier_indices']]
        
        axes[1, 0].hist(data, bins=20, alpha=0.7, color='lightcoral', label='With Outliers', density=True)
        axes[1, 0].hist(clean_data, bins=15, alpha=0.7, color='lightblue', label='Without Outliers', density=True)
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics table
        axes[1, 1].axis('off')
        
        summary_data = [
            ['Statistic', 'With Outliers', 'Without Outliers', 'Change'],
            ['Mean', f"{impact['original_mean']:.2f}", f"{impact['clean_mean']:.2f}", f"{impact['mean_change']:.2f}"],
            ['Std Dev', f"{impact['original_std']:.2f}", f"{impact['clean_std']:.2f}", f"{impact['std_change']:.2f}"],
            ['Median', f"{impact['original_median']:.2f}", f"{impact['clean_median']:.2f}", f"{impact['median_change']:.2f}"],
            ['Data Points', f"{len(data)}", f"{impact['data_points_remaining']}", f"-{impact['outliers_removed']}"]
        ]
        
        table = axes[1, 1].table(cellText=summary_data[1:], colLabels=summary_data[0],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


def demonstrate_outlier_visualization():
    """Demonstrate outlier visualization capabilities."""
    print("=== Outlier Visualization Demonstration ===\n")
    
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 50).tolist()
    outliers = [120, 150, -20, -10]
    sample_data = normal_data + outliers
    
    print(f"Sample data created: {len(sample_data)} points")
    print(f"Added outliers: {outliers}")
    print(f"Data range: {min(sample_data):.2f} to {max(sample_data):.2f}")
    print()
    
    # Initialize visualizer
    visualizer = OutlierVisualizer()
    
    print("Creating visualization plots...")
    print("1. Outlier detection methods comparison")
    visualizer.plot_outlier_detection_comparison(sample_data)
    
    print("2. Boxplot with outliers")
    visualizer.plot_boxplot_with_outliers(sample_data)
    
    print("3. Distribution with outliers (IQR method)")
    visualizer.plot_distribution_with_outliers(sample_data, method='iqr')
    
    print("4. Outlier impact analysis")
    visualizer.plot_outlier_impact_analysis(sample_data)
    
    print("Visualization demonstration completed!")


if __name__ == "__main__":
    demonstrate_outlier_visualization()

