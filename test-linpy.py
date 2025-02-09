import linpy as lp
import numpy as np

# Create a matrix
m2 = lp.Matrix([[2, 4], [6, 8]])
m3 = lp.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print(m2, m2.diagonal, m2.anti_diagonal, m2.trace, m2.anti_trace)
# print(m3, m3.diagonal, m3.anti_diagonal, m3.trace, m3.anti_trace)

# 2 2d vectorw (perpendicular)
v1 = lp.Vector([1, 0])
v2 = lp.Vector([1, 1])
print(v1.angle_between(v2))