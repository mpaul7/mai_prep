import numpy as np

# Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("a:", a)
print("b:", b)
# Dot product
dot_product = np.dot(a, b)

"""dot product of a and b
    a = [a1, a2, a3]
    b = [b1, b2, b3]
    dot product = a1*b1 + a2*b2 + a3*b3
    Args:
        a (np.ndarray): Vector a
        b (np.ndarray): Vector b
"""
    

print("Dot product of a and b:", dot_product)

# Matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A:", A)
print("B:", B)
# Matrix multiplication
matrix_mult = np.dot(A, B)

"""matrix multiplication of A and B
    A = [[a11, a12], [a21, a22]]
    B = [[b11, b12], [b21, b22]]
    matrix multiplication = [[a11*b11 + a12*b21, a11*b12 + a12*b22], [a21*b11 + a22*b21, a21*b12 + a22*b22]]
    Args:
        A (np.ndarray): Matrix A
        B (np.ndarray): Matrix B
"""
print(f"Matrix multiplication A*B:\n{matrix_mult}")

# Transpose
print(f"Transpose of A:\n{A.T}")

# Determinant
det_A = np.linalg.det(A)
print(f"Determinant of A:{det_A}")

# Inverse
inv_A = np.linalg.inv(A)
print(f"Inverse of A:\n{inv_A}")
