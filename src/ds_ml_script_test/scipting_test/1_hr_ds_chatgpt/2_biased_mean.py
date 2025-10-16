import numpy as np

population_str = "1 2 3 4 5 6 7 8 9 10"
threshold = 5
population = list(map(int, population_str.split()))
threshold = int(threshold)

population = np.array(population)

pop_mean = np.mean(population)
sample_biased = population[population > threshold]
biased_mean = np.mean(sample_biased)
bias = biased_mean - pop_mean

print(f"Population Mean: {pop_mean:.2f}")
print(f"Biased Sample Mean: {biased_mean:.2f}")
print(f"Bias: {bias:.2f}")


"""_summary_
Concept	Insight
Statistical Bias    = Difference between estimator’s expected value and true parameter
Selection Bias = Non-representative sampling → over/underestimate
Survivorship Bias = Only considering “survivors” → inflated success rate
Omitted Variable Bias = Missing key variables → misleading results
Recall / Observer / Funding Bias = Measurement or reporting errors
"""