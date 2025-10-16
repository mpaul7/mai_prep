"""
Dispersion Metrics Visualization for Data Science Interview Preparation

This module provides visualization tools for understanding dispersion metrics.
Useful for explaining concepts during interviews and understanding data spread patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Tuple
from dispersion import DispersionCalculator
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


class DispersionVisualizer:
    """
    Visualization toolkit for dispersion metrics.
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
        self.calc = DispersionCalculator()
    
    def plot_dispersion_comparison(self, datasets: List[Tuple[str, List[Union[int, float]]]], 
                                 save_path: Optional[str] = None):
        """
        Create a comprehensive comparison plot of dispersion metrics across multiple datasets.
        
        Args:
            datasets: List of tuples (name, data) for each dataset
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dispersion Metrics Comparison Across Datasets', fontsize=16, fontweight='bold')
        
        # Collect all dispersion measures
        all_measures = {}
        dataset_names = []
        
        for name, data in datasets:
            dataset_names.append(name)
            measures = self.calc.compare_dispersion_measures(data)
            
            for _, row in measures.iterrows():
                measure_name = row['measure']
                if measure_name not in all_measures:
                    all_measures[measure_name] = []
                
                # Handle both numeric and string values
                value = row['value']
                if isinstance(value, (int, float)):
                    all_measures[measure_name].append(value)
                else:
                    all_measures[measure_name].append(0)  # Error cases
        
        # Plot 1: Range comparison
        if 'Range' in all_measures:
            axes[0, 0].bar(dataset_names, all_measures['Range'], color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Range Comparison')
            axes[0, 0].set_ylabel('Range Value')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: IQR comparison
        if 'IQR' in all_measures:
            axes[0, 1].bar(dataset_names, all_measures['IQR'], color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('IQR Comparison')
            axes[0, 1].set_ylabel('IQR Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Standard Deviation comparison
        if 'Sample Std Dev' in all_measures:
            axes[0, 2].bar(dataset_names, all_measures['Sample Std Dev'], color='lightcoral', alpha=0.7)
            axes[0, 2].set_title('Standard Deviation Comparison')
            axes[0, 2].set_ylabel('Std Dev Value')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Box plots for all datasets
        data_for_boxplot = [data for _, data in datasets]
        box_plot = axes[1, 0].boxplot(data_for_boxplot, labels=dataset_names, patch_artist=True)
        
        # Color the box plots
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors[:len(datasets)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 0].set_title('Box Plot Comparison')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Histograms overlay
        for i, (name, data) in enumerate(datasets):
            axes[1, 1].hist(data, bins=15, alpha=0.6, label=name, density=True)
        
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Dispersion measures summary table
        axes[1, 2].axis('off')
        
        # Create summary table
        table_data = []
        measures_to_show = ['Range', 'IQR', 'Sample Std Dev', 'Sample Variance']
        
        for measure in measures_to_show:
            if measure in all_measures:
                row = [measure] + [f"{val:.2f}" if isinstance(val, (int, float)) else "N/A" 
                                 for val in all_measures[measure]]
                table_data.append(row)
        
        if table_data:
            table = axes[1, 2].table(cellText=table_data, 
                                   colLabels=['Measure'] + dataset_names,
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        
        axes[1, 2].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_variance_vs_std_dev(self, data: List[Union[int, float]], 
                                save_path: Optional[str] = None):
        """
        Visualize the relationship between variance and standard deviation.
        
        Args:
            data: List of numeric values
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Variance vs Standard Deviation Analysis', fontsize=16, fontweight='bold')
        
        # Calculate statistics
        var_result = self.calc.calculate_variance(data, sample=True)
        std_result = self.calc.calculate_standard_deviation(data, sample=True)
        
        # Plot 1: Data distribution
        axes[0, 0].hist(data, bins=15, alpha=0.7, color='skyblue', density=True)
        axes[0, 0].axvline(var_result['mean'], color='red', linestyle='--', 
                          label=f'Mean: {var_result["mean"]:.2f}')
        axes[0, 0].axvline(var_result['mean'] - std_result['std_dev'], color='orange', 
                          linestyle=':', label=f'Mean - 1σ')
        axes[0, 0].axvline(var_result['mean'] + std_result['std_dev'], color='orange', 
                          linestyle=':', label=f'Mean + 1σ')
        
        axes[0, 0].set_title('Data Distribution with Standard Deviation')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Deviations from mean
        mean = var_result['mean']
        deviations = [x - mean for x in data]
        squared_deviations = [(x - mean) ** 2 for x in data]
        
        x_pos = range(len(data))
        axes[0, 1].bar(x_pos, deviations, alpha=0.7, color='lightgreen', 
                      label='Deviations from Mean')
        axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].set_title('Deviations from Mean')
        axes[0, 1].set_xlabel('Data Point Index')
        axes[0, 1].set_ylabel('Deviation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Squared deviations
        axes[1, 0].bar(x_pos, squared_deviations, alpha=0.7, color='lightcoral', 
                      label='Squared Deviations')
        axes[1, 0].axhline(var_result['variance'], color='red', linestyle='--', 
                          label=f'Variance: {var_result["variance"]:.2f}')
        axes[1, 0].set_title('Squared Deviations from Mean')
        axes[1, 0].set_xlabel('Data Point Index')
        axes[1, 0].set_ylabel('Squared Deviation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Variance vs Standard Deviation relationship
        axes[1, 1].axis('off')
        
        # Create information display
        info_text = f"""
        Variance vs Standard Deviation Relationship:
        
        Data Points: {len(data)}
        Mean: {var_result['mean']:.3f}
        
        Sample Variance (σ²): {var_result['variance']:.3f}
        Sample Std Dev (σ): {std_result['std_dev']:.3f}
        
        Relationship: σ = √(σ²)
        Check: √{var_result['variance']:.3f} = {np.sqrt(var_result['variance']):.3f}
        
        Interpretation:
        • Variance is in squared units
        • Std Dev is in original units
        • Std Dev is more interpretable
        • Both measure spread around the mean
        
        Sample vs Population:
        • Sample uses n-1 denominator
        • Population uses n denominator
        • Sample variance is larger (unbiased estimator)
        """
        
        axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Variance vs Standard Deviation Info')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_robust_vs_nonrobust(self, normal_data: List[Union[int, float]], 
                                outlier_data: List[Union[int, float]], 
                                save_path: Optional[str] = None):
        """
        Compare robust vs non-robust dispersion measures.
        
        Args:
            normal_data: Data without outliers
            outlier_data: Data with outliers
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Robust vs Non-Robust Dispersion Measures', fontsize=16, fontweight='bold')
        
        # Calculate measures for both datasets
        normal_measures = self.calc.compare_dispersion_measures(normal_data)
        outlier_measures = self.calc.compare_dispersion_measures(outlier_data)
        
        # Extract robust and non-robust measures
        robust_measures = ['IQR', 'MAD (from median)']
        nonrobust_measures = ['Range', 'Sample Std Dev', 'Sample Variance']
        
        # Plot 1: Data comparison
        axes[0, 0].hist(normal_data, bins=10, alpha=0.7, label='Normal Data', color='skyblue')
        axes[0, 0].hist(outlier_data, bins=15, alpha=0.7, label='Data with Outliers', color='lightcoral')
        axes[0, 0].set_title('Data Distribution Comparison')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot comparison
        box_data = [normal_data, outlier_data]
        box_labels = ['Normal', 'With Outliers']
        box_plot = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
        
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0, 1].set_title('Box Plot Comparison')
        axes[0, 1].set_ylabel('Value')
        
        # Plot 3: Robust measures comparison
        robust_normal = []
        robust_outlier = []
        robust_names = []
        
        for measure in robust_measures:
            normal_val = normal_measures[normal_measures['measure'] == measure]['value'].iloc[0]
            outlier_val = outlier_measures[outlier_measures['measure'] == measure]['value'].iloc[0]
            
            if isinstance(normal_val, (int, float)) and isinstance(outlier_val, (int, float)):
                robust_normal.append(normal_val)
                robust_outlier.append(outlier_val)
                robust_names.append(measure)
        
        x_pos = np.arange(len(robust_names))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, robust_normal, width, label='Normal Data', 
                      color='skyblue', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, robust_outlier, width, label='With Outliers', 
                      color='lightcoral', alpha=0.7)
        
        axes[1, 0].set_title('Robust Measures Comparison')
        axes[1, 0].set_xlabel('Measure')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(robust_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Non-robust measures comparison
        nonrobust_normal = []
        nonrobust_outlier = []
        nonrobust_names = []
        
        for measure in nonrobust_measures:
            normal_val = normal_measures[normal_measures['measure'] == measure]['value'].iloc[0]
            outlier_val = outlier_measures[outlier_measures['measure'] == measure]['value'].iloc[0]
            
            if isinstance(normal_val, (int, float)) and isinstance(outlier_val, (int, float)):
                nonrobust_normal.append(normal_val)
                nonrobust_outlier.append(outlier_val)
                nonrobust_names.append(measure)
        
        x_pos = np.arange(len(nonrobust_names))
        
        axes[1, 1].bar(x_pos - width/2, nonrobust_normal, width, label='Normal Data', 
                      color='skyblue', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, nonrobust_outlier, width, label='With Outliers', 
                      color='lightcoral', alpha=0.7)
        
        axes[1, 1].set_title('Non-Robust Measures Comparison')
        axes[1, 1].set_xlabel('Measure')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(nonrobust_names, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_coefficient_of_variation_comparison(self, datasets: List[Tuple[str, List[Union[int, float]]]], 
                                               save_path: Optional[str] = None):
        """
        Compare coefficient of variation across datasets with different scales.
        
        Args:
            datasets: List of tuples (name, data) for each dataset
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Coefficient of Variation Analysis', fontsize=16, fontweight='bold')
        
        # Calculate CV for each dataset
        cv_data = []
        means = []
        std_devs = []
        
        for name, data in datasets:
            try:
                cv_result = self.calc.calculate_coefficient_of_variation(data, sample=True)
                cv_data.append((name, cv_result['cv_percent'], cv_result['mean'], cv_result['std_dev']))
                means.append(cv_result['mean'])
                std_devs.append(cv_result['std_dev'])
            except Exception as e:
                print(f"Error calculating CV for {name}: {e}")
        
        # Plot 1: Data distributions
        for name, data in datasets:
            axes[0, 0].hist(data, bins=10, alpha=0.6, label=name, density=True)
        
        axes[0, 0].set_title('Data Distributions (Normalized)')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Means comparison
        names = [item[0] for item in cv_data]
        dataset_means = [item[2] for item in cv_data]
        
        axes[0, 1].bar(names, dataset_means, color='lightblue', alpha=0.7)
        axes[0, 1].set_title('Means Comparison')
        axes[0, 1].set_ylabel('Mean Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(dataset_means):
            axes[0, 1].text(i, v + max(dataset_means)*0.01, f'{v:.2f}', 
                           ha='center', va='bottom')
        
        # Plot 3: Standard deviations comparison
        dataset_stds = [item[3] for item in cv_data]
        
        axes[1, 0].bar(names, dataset_stds, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Standard Deviations Comparison')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(dataset_stds):
            axes[1, 0].text(i, v + max(dataset_stds)*0.01, f'{v:.2f}', 
                           ha='center', va='bottom')
        
        # Plot 4: Coefficient of variation comparison
        dataset_cvs = [item[1] for item in cv_data]
        
        bars = axes[1, 1].bar(names, dataset_cvs, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('Coefficient of Variation Comparison')
        axes[1, 1].set_ylabel('CV (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels and interpretation
        for i, (bar, cv) in enumerate(zip(bars, dataset_cvs)):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(dataset_cvs)*0.01,
                           f'{cv:.1f}%', ha='center', va='bottom')
            
            # Color code based on CV interpretation
            if cv < 15:
                bar.set_color('lightgreen')
            elif cv < 35:
                bar.set_color('lightyellow')
            else:
                bar.set_color('lightcoral')
        
        # Add interpretation legend
        axes[1, 1].text(0.02, 0.98, 'CV Interpretation:\nGreen: Low variability (<15%)\nYellow: Moderate (15-35%)\nRed: High variability (>35%)', 
                       transform=axes[1, 1].transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


def demonstrate_dispersion_visualization():
    """Demonstrate dispersion visualization capabilities."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot run visualization demonstration.")
        return
    
    print("=== Dispersion Visualization Demonstration ===\n")
    
    # Create sample datasets
    np.random.seed(42)
    
    # Dataset 1: Low variability
    low_var_data = np.random.normal(50, 5, 100).tolist()
    
    # Dataset 2: High variability
    high_var_data = np.random.normal(50, 15, 100).tolist()
    
    # Dataset 3: With outliers
    outlier_data = np.random.normal(50, 8, 95).tolist() + [100, 120, 130, 140, 150]
    
    # Dataset 4: Different scale, similar relative variability
    different_scale = [x * 10 for x in np.random.normal(5, 0.5, 100)]
    
    datasets = [
        ("Low Variability", low_var_data),
        ("High Variability", high_var_data),
        ("With Outliers", outlier_data),
        ("Different Scale", different_scale)
    ]
    
    print(f"Created {len(datasets)} datasets for demonstration")
    for name, data in datasets:
        print(f"{name}: mean={np.mean(data):.2f}, std={np.std(data, ddof=1):.2f}, n={len(data)}")
    print()
    
    # Initialize visualizer
    visualizer = DispersionVisualizer()
    
    print("Creating visualization plots...")
    
    print("1. Dispersion comparison across datasets")
    visualizer.plot_dispersion_comparison(datasets)
    
    print("2. Variance vs Standard Deviation analysis")
    visualizer.plot_variance_vs_std_dev(high_var_data)
    
    print("3. Robust vs Non-robust measures comparison")
    normal_data = low_var_data[:50]  # First 50 points without outliers
    outlier_data_subset = outlier_data  # Data with outliers
    visualizer.plot_robust_vs_nonrobust(normal_data, outlier_data_subset)
    
    print("4. Coefficient of Variation comparison")
    cv_datasets = [
        ("Small Scale", [10, 11, 12, 13, 14]),
        ("Medium Scale", [100, 110, 120, 130, 140]),
        ("Large Scale", [1000, 1100, 1200, 1300, 1400]),
        ("High Variability", [50, 70, 90, 110, 130])
    ]
    visualizer.plot_coefficient_of_variation_comparison(cv_datasets)
    
    print("Dispersion visualization demonstration completed!")


if __name__ == "__main__":
    demonstrate_dispersion_visualization()
