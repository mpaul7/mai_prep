import numpy as np

# Example data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
input_str = "1 2 3 4 5 6 7 8 9 10"
print(input_str.split())
data = list(map(float, input_str.split()))

print(data)

mean = np.mean(data)
print(round(mean, 2))

print(mean)