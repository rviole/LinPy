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

# Test rank property
print("Rank of m3:", m3.rank)  # Should be 2
print("Rank of m4:", m4.rank)  # Should be 3

# Test is_full_rank property
print("Is m3 full rank:", m3.is_full_rank)  # Should be False
print("Is m4 full rank:", m4.is_full_rank)  # Should be True

# Test determinant property
print("Determinant of m3:", m3.determinant)  # Should be 0
print("Determinant of m4:", m4.determinant)  # Should be 45

# Test inverse property
try:
    print("Inverse of m4:\n", m4.inverse)  # Should print the inverse of m4
except ValueError as e:
    print(e)

try:
    print("Inverse of m3:\n", m3.inverse)  # Should raise an error
except ValueError as e:
    print(e)
  
# Test is_singular property
print("Is m3 singular:", m3.is_singular)  # Should be True
print("Is m4 singular:", m4.is_singular)  # Should be False
 
# Test is_invertable property
print("Is m3 invertable:", m3.is_invertable)  # Should be False
print("Is m4 invertable:", m4.is_invertable)  # Should be True

# Test is_linear_transformation property
print("Is m3 a linear transformation:", m3.is_linear_transformation)  # Should be False
print("Is m4 a linear transformation:", m4.is_linear_transformation)  # Should be True

v1 = lp.Vector([1,2,3])
v2 = lp.Vector([2, 3, 4])
print(v1 @ v2)
print(v1 * v2)


v1 = lp.Vector([1, 0 ,0])
v2 = lp.Vector([0, 1 ,0])
v3 = lp.Vector([0, 0 ,1])
v4 = lp.Vector([1, 1 ,1])
basis = lp.matrix_from_vectors(v1, v2)
print(basis)
print(basis.is_basis)

print(basis.is_linear_combination(v3))

