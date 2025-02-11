import linpy as lp
import numpy as np

# Create a matrix
m1 = lp.Matrix([[1], [3]])
m2 = lp.Matrix([[2, 4], [6, 8]])
m3 = lp.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# upper triangular matrix
m4 = lp.Matrix([[1, 2, 3], [0, 5, 6], [0, 0, 9]])

# lower triangular matrix
m5 = lp.Matrix([[1, 0, 0], [4, 5, 0], [7, 8, 9]])

# diagonal matrix
m6 = lp.Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 9]])

# identity matrix
m7 = lp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

print(m4.is_full_rank)
print(m5.is_full_rank)
print(m6.is_full_rank)
print(m7.is_full_rank)