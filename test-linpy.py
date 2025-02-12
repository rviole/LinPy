import linpy as lp
import numpy as np

# Create vectors
v1 = lp.Vector([1, 2, 3])
v2 = lp.Vector([4, 5, 6])

# Create matrices
m1 = lp.Matrix([[1, 2], [3, 4]])
m2 = lp.Matrix([[5, 6], [7, 8]])

# Create vectors from NumPy arrays
import numpy as np

arr = np.array([1, 2, 3])
v3 = lp.Vector(arr)

# Create matrices from NumPy arrays
arr2 = np.array([[1, 2], [3, 4]])
m3 = lp.Matrix(arr2)