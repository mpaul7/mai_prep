"""
Advanced Sampling Practice Problems for Data Science Interviews

This module contains more challenging problems commonly found in technical interviews
and coding assessments for data science positions.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
import math
from sampling import SamplingToolkit


def advanced_sampling_problems():
    """
    Collection of advanced sampling problems for interview preparation.
    """
    
    def problem_5_bootstrap_sampling():
        """
        Problem 5: Bootstrap Sampling
        
        Implement bootstrap sampling to estimate the sampling distribution of the mean.
        Generate n bootstrap samples and return the mean of bootstrap means.
        
        Input: sample = [1, 2, 3, 4, 5], n_bootstrap = 1000
        """
        def bootstrap_mean_estimate(sample: List[Union[int, float]], 
                                  n_bootstrap: int = 1000) -> float:
            toolkit = SamplingToolkit(random_seed=42)
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                bootstrap_sample = toolkit.sample_with_replacement(sample, len(sample))
                bootstrap_mean = toolkit.calculate_sample_mean(bootstrap_sample)
                bootstrap_means.append(bootstrap_mean)
            
            return round(sum(bootstrap_means) / len(bootstrap_means), 4)
        
        # Test case
        test_sample = [1, 2, 3, 4, 5]
        result = bootstrap_mean_estimate(test_sample, 1000)
        original_mean = sum(test_sample) / len(test_sample)
        
        print(f"Problem 5 - Original Sample: {test_sample}")
        print(f"Problem 5 - Original Mean: {original_mean}")
        print(f"Problem 5 - Bootstrap Mean Estimate: {result}")
        return result
    
    def problem_6_confidence_interval():
        """
        Problem 6: Confidence Interval for Sample Mean
        
        Calculate the 95% confidence interval for the sample mean using bootstrap.
        
        Input: sample = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        """
        def confidence_interval_bootstrap(sample: List[Union[int, float]], 
                                        confidence_level: float = 0.95,
                                        n_bootstrap: int = 1000) -> Tuple[float, float]:
            toolkit = SamplingToolkit(random_seed=42)
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                bootstrap_sample = toolkit.sample_with_replacement(sample, len(sample))
                bootstrap_mean = toolkit.calculate_sample_mean(bootstrap_sample)
                bootstrap_means.append(bootstrap_mean)
            
            bootstrap_means.sort()
            alpha = 1 - confidence_level
            lower_idx = int((alpha / 2) * n_bootstrap)
            upper_idx = int((1 - alpha / 2) * n_bootstrap)
            
            return (round(bootstrap_means[lower_idx], 2), 
                   round(bootstrap_means[upper_idx], 2))
        
        # Test case
        test_sample = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        ci_lower, ci_upper = confidence_interval_bootstrap(test_sample)
        sample_mean = sum(test_sample) / len(test_sample)
        
        print(f"Problem 6 - Sample: {test_sample}")
        print(f"Problem 6 - Sample Mean: {sample_mean}")
        print(f"Problem 6 - 95% Confidence Interval: ({ci_lower}, {ci_upper})")
        return (ci_lower, ci_upper)
    
    def problem_7_sample_size_calculation():
        """
        Problem 7: Sample Size Calculation
        
        Given a population standard deviation and desired margin of error,
        calculate the required sample size for a given confidence level.
        
        Formula: n = (Z * σ / E)²
        Where Z is the Z-score, σ is population std dev, E is margin of error
        """
        def calculate_sample_size(population_std: float, 
                                margin_of_error: float,
                                confidence_level: float = 0.95) -> int:
            # Z-scores for common confidence levels
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            
            if confidence_level not in z_scores:
                raise ValueError("Confidence level must be 0.90, 0.95, or 0.99")
            
            z_score = z_scores[confidence_level]
            sample_size = (z_score * population_std / margin_of_error) ** 2
            
            return math.ceil(sample_size)
        
        # Test case
        pop_std = 15.0
        margin_error = 3.0
        conf_level = 0.95
        
        required_n = calculate_sample_size(pop_std, margin_error, conf_level)
        
        print(f"Problem 7 - Population Std Dev: {pop_std}")
        print(f"Problem 7 - Margin of Error: {margin_error}")
        print(f"Problem 7 - Confidence Level: {conf_level}")
        print(f"Problem 7 - Required Sample Size: {required_n}")
        return required_n
    
    def problem_8_sampling_distribution():
        """
        Problem 8: Central Limit Theorem Demonstration
        
        Sample from a non-normal distribution and show that sample means
        approach normal distribution.
        """
        def demonstrate_clt(population_size: int = 10000, 
                          sample_size: int = 30,
                          n_samples: int = 1000) -> Dict[str, float]:
            # Create a skewed population (exponential distribution)
            np.random.seed(42)
            population = np.random.exponential(scale=2, size=population_size)
            
            toolkit = SamplingToolkit(random_seed=42)
            sample_means = []
            
            for _ in range(n_samples):
                sample = toolkit.simple_random_sample(population.tolist(), sample_size)
                sample_mean = toolkit.calculate_sample_mean(sample)
                sample_means.append(sample_mean)
            
            return {
                'population_mean': round(float(np.mean(population)), 4),
                'population_std': round(float(np.std(population)), 4),
                'sample_means_mean': round(float(np.mean(sample_means)), 4),
                'sample_means_std': round(float(np.std(sample_means)), 4),
                'theoretical_std_error': round(float(np.std(population) / np.sqrt(sample_size)), 4)
            }
        
        # Test case
        results = demonstrate_clt()
        
        print(f"Problem 8 - Central Limit Theorem Demonstration:")
        print(f"Population Mean: {results['population_mean']}")
        print(f"Population Std Dev: {results['population_std']}")
        print(f"Sample Means Mean: {results['sample_means_mean']}")
        print(f"Sample Means Std Dev: {results['sample_means_std']}")
        print(f"Theoretical Standard Error: {results['theoretical_std_error']}")
        return results
    
    def problem_9_stratified_sampling_analysis():
        """
        Problem 9: Stratified Sampling Efficiency
        
        Compare simple random sampling vs stratified sampling for a given dataset.
        """
        def compare_sampling_methods(data: pd.DataFrame, 
                                   target_column: str,
                                   strata_column: str,
                                   sample_size: int = 100) -> Dict[str, float]:
            toolkit = SamplingToolkit(random_seed=42)
            
            # Simple random sampling
            simple_sample = data.sample(n=sample_size, random_state=42)
            simple_mean = simple_sample[target_column].mean()
            
            # Stratified sampling
            stratified_sample = toolkit.stratified_sample(data, strata_column, sample_size)
            stratified_mean = stratified_sample[target_column].mean()
            
            # Population mean
            population_mean = data[target_column].mean()
            
            return {
                'population_mean': round(population_mean, 4),
                'simple_sample_mean': round(simple_mean, 4),
                'stratified_sample_mean': round(stratified_mean, 4),
                'simple_error': round(abs(simple_mean - population_mean), 4),
                'stratified_error': round(abs(stratified_mean - population_mean), 4)
            }
        
        # Create test dataset
        np.random.seed(42)
        data = pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(10, 2, 300),  # Group A
                np.random.normal(20, 3, 200),  # Group B
                np.random.normal(30, 1, 100)   # Group C
            ]),
            'group': ['A'] * 300 + ['B'] * 200 + ['C'] * 100
        })
        
        results = compare_sampling_methods(data, 'value', 'group', 60)
        
        print(f"Problem 9 - Sampling Methods Comparison:")
        print(f"Population Mean: {results['population_mean']}")
        print(f"Simple Random Sample Mean: {results['simple_sample_mean']} (Error: {results['simple_error']})")
        print(f"Stratified Sample Mean: {results['stratified_sample_mean']} (Error: {results['stratified_error']})")
        return results
    
    def problem_10_outlier_impact():
        """
        Problem 10: Impact of Outliers on Sample Mean
        
        Analyze how outliers affect sample mean calculations.
        """
        def analyze_outlier_impact(base_sample: List[Union[int, float]],
                                 outlier_values: List[Union[int, float]]) -> Dict[str, float]:
            toolkit = SamplingToolkit()
            
            # Original sample statistics
            original_mean = toolkit.calculate_sample_mean(base_sample)
            original_stats = toolkit.calculate_sample_statistics(base_sample)
            
            # Sample with outliers
            sample_with_outliers = base_sample + outlier_values
            outlier_mean = toolkit.calculate_sample_mean(sample_with_outliers)
            outlier_stats = toolkit.calculate_sample_statistics(sample_with_outliers)
            
            return {
                'original_mean': round(original_mean, 4),
                'original_std': round(original_stats['std_dev'], 4),
                'outlier_mean': round(outlier_mean, 4),
                'outlier_std': round(outlier_stats['std_dev'], 4),
                'mean_change': round(outlier_mean - original_mean, 4),
                'std_change': round(outlier_stats['std_dev'] - original_stats['std_dev'], 4)
            }
        
        # Test case
        base_sample = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        outliers = [100, 150]
        
        results = analyze_outlier_impact(base_sample, outliers)
        
        print(f"Problem 10 - Outlier Impact Analysis:")
        print(f"Original Sample: {base_sample}")
        print(f"Outliers: {outliers}")
        print(f"Original Mean: {results['original_mean']} (Std: {results['original_std']})")
        print(f"With Outliers Mean: {results['outlier_mean']} (Std: {results['outlier_std']})")
        print(f"Mean Change: {results['mean_change']}, Std Change: {results['std_change']}")
        return results
    
    # Run all advanced problems
    print("=== Advanced Sampling Problems ===\n")
    problem_5_bootstrap_sampling()
    print()
    problem_6_confidence_interval()
    print()
    problem_7_sample_size_calculation()
    print()
    problem_8_sampling_distribution()
    print()
    problem_9_stratified_sampling_analysis()
    print()
    problem_10_outlier_impact()


if __name__ == "__main__":
    advanced_sampling_problems()
