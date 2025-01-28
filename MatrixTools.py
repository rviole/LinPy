import numpy as np
from BaseTools import (
    Matrix,
    Vector,
    validate_input,
    validate_multiplication_compatibility,
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

    validate_multiplication_compatibility(A, vector)

    return A @ vector


def compose_transformations(vector, *matrices):
    validate_input(vector, *matrices)

    # validate types
    if not isinstance(vector, Vector):
        vector = Vector(vector)
    matrices = [Matrix(m) if not isinstance(m, Matrix) else m for m in matrices]

    # validate compatibility between matrices and a vector
    for matrix in matrices:
        validate_multiplication_compatibility(matrix, vector, raise_exception=True)

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


def matrix_multiply(matrix1, matrix2):
    validate_input(matrix1, matrix2)
    matrix1 = Matrix(matrix1) if not isinstance(matrix1, Matrix) else matrix1
    matrix2 = Matrix(matrix2) if not isinstance(matrix2, Matrix) else matrix2
    validate_multiplication_compatibility(matrix1, matrix2)

    return matrix1 @ matrix2


def transpose(matrix):
    validate_input(matrix)
    matrix = Matrix(matrix) if not isinstance(matrix, Matrix) else matrix

    n_rows, n_cols = matrix.shape[:2]
    new_n_rows = n_cols
    new_n_cols = n_rows

    new_matrix = np.zeros((new_n_rows, new_n_cols))
    for idx in range(new_n_rows):
        # go through empty new matrix rows, and assign cols of the original matrix to them
        new_matrix[idx] = matrix[:, idx]

    return Matrix(new_matrix)


def calculate_inverse_matrix(matrix):
    validate_input(matrix)
    matrix = Matrix(matrix) if not isinstance(matrix, Matrix) else matrix

    if not matrix.is_square():
        raise ValueError("Only square matrices can be inverted.")

    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        raise ValueError("Matrix is singular and cannot be inverted.")

    return Matrix(inv_matrix)


def calculate_determinant(matrix):
    validate_input(matrix)
    matrix = Matrix(matrix) if not isinstance(matrix, Matrix) else matrix

    if not matrix.is_square():
        raise ValueError("Only square matrices have a determinant.")

    if matrix.shape == (1, 1):
        determinant = matrix[0, 0]
    elif matrix.shape == (2, 2):
        determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    elif matrix.shape == (3, 3):
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        determinant = (
            (a * e * i)
            + (b * f * g)
            + (c * d * h)
            - (c * e * g)
            - (b * d * i)
            - (a * f * h)
        )
    else:
        determinant = np.linalg.det(matrix)
    return determinant.astype(np.float64)


def calculate_trace(matrix):
    validate_input(matrix)
    matrix = Matrix(matrix) if not isinstance(matrix, Matrix) else matrix

    if not matrix.is_square():
        raise ValueError("Only square matrices have a trace.")

    n_rows = matrix.shape[0]
    trace = sum(matrix[i, i] for i in range(n_rows))
    return trace.astype(np.float64)


def calculate_diagonal_sum(matrix):
    return calculate_trace(matrix)


def calculate_antidiagonal_sum(matrix):
    validate_input(matrix)
    matrix = Matrix(matrix) if not isinstance(matrix, Matrix) else matrix

    if not matrix.is_square():
        raise ValueError("Only square matrices have an antidiagonal sum.")

    n_rows = matrix.shape[0]
    antidiagonal_sum = sum(matrix[i, n_rows - 1 - i] for i in range(n_rows))
    return antidiagonal_sum.astype(np.float64)


def find_diagonal(matrix):
    validate_input(matrix)
    matrix = Matrix(matrix) if not isinstance(matrix, Matrix) else matrix

    if not matrix.is_square():
        raise ValueError("Only square matrices have a diagonal.")

    n_rows = matrix.shape[0]
    diagonal = [matrix[i, i] for i in range(n_rows)]
    return Vector(diagonal)


def find_anti_diagonal(matrix):
    validate_input(matrix)
    matrix = Matrix(matrix) if not isinstance(matrix, Matrix) else matrix

    if not matrix.is_square():
        raise ValueError("Only square matrices have an anti-diagonal.")

    n_rows = matrix.shape[0]
    anti_diagonal = [matrix[i, n_rows - 1 - i] for i in range(n_rows)]
    return Vector(anti_diagonal)


# need to add corresponding tests
# update Matrix class to include these methods