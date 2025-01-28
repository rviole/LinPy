import numpy as np
from BaseTools import (
    Matrix,
    Vector,
    validate_input,
    validate_matrix_vector_compatibility,
)


def is_square(*matrices) -> bool:
    matrices = [Matrix(m) if not isinstance(m, Matrix) else m for m in matrices]
    return all(m.shape[0] == m.shape[1] for m in matrices)


def apply_transformation(transformation_matrix, vector):

    A = transformation_matrix
    validate_input(A, vector)

    if not isinstance(A, Matrix):
        A = Matrix(A)
    if not isinstance(vector, Vector):
        vector = Vector(vector)

    validate_matrix_vector_compatibility(A, vector)

    return A @ vector


def compose_transformations(vector, *matrices):
    validate_input(vector, *matrices)

    # validate types
    if not isinstance(vector, Vector):
        vector = Vector(vector)
    matrices = [Matrix(m) if not isinstance(m, Matrix) else m for m in matrices]

    # validate compatibility between matrices and a vector
    for matrix in matrices:
        validate_matrix_vector_compatibility(matrix, vector, raise_exception=True)

    result = vector
    for matrix in matrices:
        result = matrix @ result

    return result


def is_linear_transformation(transformation_matrix) -> bool:
    A = transformation_matrix
    validate_input(A)

    if not isinstance(A, Matrix):
        A = Matrix(A)
    n_vectors = A.shape[1]

    np.random.seed(42)  # Sets the seed
    # the number of rows of each vector should be equal to the number of cols of matrix (n vectors)
    v1 = Vector(np.random.random(size=(n_vectors,)))
    v2 = Vector(np.random.random(size=(n_vectors,)))
    c = np.random.rand()

    additivity = np.allclose(A @ v1 + A @ v2, A @ (v1 + v2))
    homogeneity = np.allclose(A @ (c * v1), c * (A @ v1))
    return additivity and homogeneity
