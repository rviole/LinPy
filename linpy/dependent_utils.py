from .Vector import Vector
from .Matrix import Matrix
from typing import List

# Utils that use Vector and Matrix classes


def matrix_from_vectors(*vectors: List[Vector]) -> Matrix:
    if not all(isinstance(vector, Vector) for vector in vectors):
        raise TypeError("All elements must be instances of Vector class")
        
    return Matrix([vector.data for vector in vectors]).T


def apply_transformation_on_vector(matrix: Matrix, vector: Vector):

    if not isinstance(matrix, Matrix):
        raise TypeError("Matrix must be an instance of Matrix class")
    if not isinstance(vector, Vector):
        raise TypeError("Vector must be an instance of Vector class")

    # ensure shape compability (m, n) * (n,)
    if matrix.shape[1] != vector.shape[0]:
        raise Exception(
            f"Shapes must match (m, n) * (n,), got: {matrix.shape} * {vector.shape}"
        )

    # each column is a transformed basis vector in the matrix, so we will Transpose it for caluclation purposes
    matrix = matrix.T

    transformed_basis_vectors = []
    for i, vector_element in enumerate(vector):
        basis_vector = matrix[i]
        val = [vector_element * x for x in basis_vector]
        transformed_basis_vectors.append(val)

    return Vector([sum(x) for x in zip(*transformed_basis_vectors)])


