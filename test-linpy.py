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

# print(m2, m2.diagonal, m2.anti_diagonal, m2.trace, m2.anti_trace)
# print(m3, m3.diagonal, m3.anti_diagonal, m3.trace, m3.anti_trace)

# 2 2d vectorw (perpendicular)
v1 = lp.Vector([1, 0])
v2 = lp.Vector([1, 1])
# print(v1.angle_between(v2))

# There are 2 types of triangular matrix: upper triangular and lower triangular

# print(m4.is_identity)
# print(m5.is_identity)
# print(m6.is_identity)
# print(m7.is_identity)

#non symmetric matrix
m6 = lp.Matrix([[1, 2, 3], 
                [4, 5, 6],
                [7, 8, 9]])

# symmetric matrix
m8 = lp.Matrix([[1, 2, 3], 
                [2, 4, 5],
                [3, 5, 6]])
print(m6.is_symmetric)
print(m7.is_symmetric)
print(m8.is_symmetric)