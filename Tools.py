import numpy as np
from typing import Optional


class Vector(np.ndarray):
    """
    A class representing a mathematical vector that inherits from numpy.ndarray.

    This ensures that the Vector behaves like a numpy array and supports all
    operations such as addition, multiplication, etc.

    Attributes:
        data (numpy.ndarray): The data of the vector as a 1D numpy array.
    """

    def __new__(cls, data):
        # Convert input data into a numpy array and ensure it's a 1D vector
        obj = np.asarray(data).squeeze()  # Remove any singleton dimensions

        # If it's a 2D column vector, flatten it to 1D
        if obj.ndim == 2 and obj.shape[1] == 1:
            obj = obj.flatten()

        # If it's not a 1D array after squeezing, raise an error
        if obj.ndim != 1:
            raise ValueError("A vector must be a 1D array.")

        # Return a new object with the desired class (Vector)
        return obj.view(cls)

    def __repr__(self):
        return f"Vector({super().__repr__()})"


class Matrix(np.ndarray):
    """
    A class representing a mathematical matrix that inherits from numpy.ndarray.

    This ensures that the Matrix behaves like a numpy array and supports all
    operations such as addition, multiplication, etc.

    Attributes:
        data (numpy.ndarray): The data of the matrix as a 2D numpy array.
    """

    def __new__(cls, data):
        # Convert input data into a numpy array and ensure it's 2D
        obj = np.asarray(data).view(cls)
        if obj.ndim not in [1, 2]:
            raise ValueError("A matrix must be either a 1D or 2D array.")

        # Handle 1D array by reshaping it into a 1xN matrix
        if obj.ndim == 1:
            obj = obj.reshape(1, -1)

        # Handle single-column matrix to row vector conversion
        if obj.ndim == 2 and obj.shape[1] == 1:
            obj = obj.T

        return obj

    def __repr__(self):
        return f"Matrix({super().__repr__()})"

    def rank(self):
        return np.linalg.matrix_rank(self)


def is_dependent(*args) -> bool:
    if not args:
        raise ValueError("No input provided.")

    if len(args) == 1:
        # a  matrix
        matrix = args[0]
        matrix = Matrix(matrix) if not isinstance(matrix, Matrix) else matrix

    else:
        vectors = [Vector(v) if not isinstance(v, Vector) else v for v in args]
        matrix = Matrix(np.array(vectors))

    rank = np.linalg.matrix_rank(matrix)
    vector_num = matrix.shape[1]  # col num = vector num
    # print("Matrix:", matrix)
    # print("Rank:", rank)
    # print("Shape:", matrix.shape)
    # print("Dims:", matrix.ndim)
    # print("Cols:", vector_num)
    return rank < vector_num


def is_linear_transformation(
    transformation_matrix: Matrix,
    v1: Optional[Vector] = None,
    v2: Optional[Vector] = None,
    c: Optional[float] = None,
) -> bool:
    # generate 2 random 2D vectors
    if not (v1 and v2):
        v1, v2 = np.random.random(size=(2, 2))
        v1, v2 = Vector(v1), Vector(v2)
    else:
        if not isinstance(v1, Vector):
            v1 = Vector(v1)
        elif not isinstance(v2, Vector):
            v2 = Vector(v2)
    # random scalar
    c = np.random.rand()

    A = (
        Matrix(transformation_matrix)
        if not isinstance(transformation_matrix, Matrix)
        else transformation_matrix
    )

    additivity = np.allclose(A @ (v1 + v2), A @ v1 + A @ v2)
    homogeneity = np.allclose(A @ (c * v1), c * (A @ v1))

    print(additivity, homogeneity)
    return additivity and homogeneity


is_linear_transformation([[-1, 0], [0, 1]])
