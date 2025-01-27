import numpy as np
from typing import Optional
from BaseTools import (
    Vector,
    validate_input,
    validate_equal_shapes,
)


def is_dependent(*vectors) -> bool:

    validate_input(vectors)
    validate_equal_shapes(*vectors)  # Check if all vectors have the same shape

    vectors = [Vector(v) if not isinstance(v, Vector) else v for v in vectors]
    matrix = np.column_stack(vectors)

    rank = np.linalg.matrix_rank(matrix)
    vector_num = matrix.shape[1]  # col num = vector num

    return bool(rank < vector_num)


def find_linear_combination(target_vector, *vectors) -> Optional[np.ndarray]:
    validate_input(target_vector)
    validate_input(vectors)

    validate_equal_shapes(*vectors)  # Check if all vectors have the same shape
    validate_equal_shapes(
        target_vector, vectors[0]
    )  # Check if the target vector has the same shape as the basis vectors

    vectors = [Vector(v) if not isinstance(v, Vector) else v for v in vectors]
    matrix = np.column_stack(vectors)
    try:
        coefficients, residuals, rank, s = np.linalg.lstsq(
            matrix, target_vector, rcond=None
        )
        # If the residuals are sufficiently small, return the coefficients
        if np.allclose(
            np.dot(matrix, coefficients), target_vector
        ):  # Check if A*c is close to b
            return coefficients
        else:
            return None  # No valid linear combination
    except np.linalg.LinAlgError as e:
        print(f"An error occurred: {e}")
        return None  # In case of any numerical errors


def find_span(*basis_vectors) -> dict:
    validate_input(basis_vectors)
    validate_equal_shapes(*basis_vectors)  # Check if all vectors have the same shape

    vectors = [Vector(v) if not isinstance(v, Vector) else v for v in basis_vectors]
    matrix = np.column_stack(vectors)

    output = {
        "rank": 0,  # Dimension of the span (rank)
        "full_span": False,  # Whether the span fills the entire space
        "zero_span": False,  # Whether the span is trivial (zero vector only)
    }

    rank = np.linalg.matrix_rank(matrix)
    output["rank"] = int(rank)

    n_vectors = matrix.shape[1]  # Number of vectors is equal to number of columns

    if rank == 0:
        output["zero_span"] = True
    elif rank == n_vectors:
        output["full_span"] = True

    return output


def is_basis(*vectors) -> bool:
    validate_input(vectors)
    validate_equal_shapes(*vectors)  # Check if all vectors have the same shape

    return not is_dependent(*vectors)

