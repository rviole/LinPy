import numpy as np

np.random.seed(42)


class Matrix(np.ndarray):
    """
    A class representing a mathematical matrix that inherits from numpy.ndarray.

    This ensures that the Matrix behaves like a numpy array and supports all
    operations such as addition, multiplication, etc.

    Attributes:
        data (numpy.ndarray): The data of the matrix as a 2D numpy array.
    """

    def __new__(cls, data, make_from_vectors=False):

        if make_from_vectors:
            data = np.column_stack(data)
        else:
            obj = np.asarray(data)

        if obj.ndim == 2:
            raise ValueError(f"A matrix must be a 2D array, got {obj.ndim}D.")

        return obj.view(cls)

    def rank(self):
        base_matrix = self
        return np.linalg.matrix_rank(base_matrix)

    def is_square(self):
        base_matrix = self
        return base_matrix.shape[0] == base_matrix.shape[1]


def is_linear_transformation(transformation_matrix) -> bool:
    A = transformation_matrix
    if not isinstance(A, Matrix):
        A = Matrix(A)
    n_vectors = A.shape[1]

    # the number of rows of each vector should be equal to the number of cols of matrix (n vectors)
    v1 = np.random.random(size=(n_vectors,))
    v2 = np.random.random(size=(n_vectors,))
    c = np.random.rand()

    additivity = np.allclose(A @ v1 + A @ v2, A @ (v1 + v2))
    homogeneity = np.allclose(A @ (c * v1), c * (A @ v1))
    return additivity and homogeneity
