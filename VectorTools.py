import numpy as np
from typing import Optional
from IPython.display import display, Math


def validate_input(input_data, raise_exception=True) -> bool:
    if not input_data:
        if raise_exception:
            raise ValueError("No input provided.")
        return False
    return True


def validate_equal_shapes(*vectors, raise_exception=True) -> bool:
    if not vectors:
        raise ValueError("No input provided.")

    vectors = [Vector(v) if not isinstance(v, Vector) else v for v in vectors]
    if all(v.shape == vectors[0].shape for v in vectors):
        return True
    else:
        if raise_exception:
            raise ValueError("Shapes of all vectors must be equal.")
        return False


def decorator_validate_inputs(func):
    def wrapper(*args, **kwargs):
        if not args:
            raise ValueError("No input provided.")
        return func(*args, **kwargs)

    return wrapper


def decorator_validate_shapes(func):
    def wrapper(*args, **kwargs):
        validate_equal_shapes(*args)
        return func(*args, **kwargs)

    return wrapper


class Vector(np.ndarray):
    """
    A class representing a mathematical vector that inherits from numpy.ndarray.

    This ensures that the Vector behaves like a numpy array and supports all
    operations such as addition, multiplication, etc.

    Attributes:
        data (numpy.ndarray): The data of the vector as a 1D numpy array.
    """

    def __new__(cls, data):

        try:
            import IPython

            get_ipython()  # type: ignore
            cls.ipython_env = True
        except:
            cls.ipython_env = False

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
        # If there are more than 5 elements, format it with first 2, ellipsis, and last 2
        if self.ipython_env:

            if len(self) > 5:
                latex_repr = (
                    r"Vector\left(\begin{bmatrix} "
                    + " \\\\ ".join(map(str, self[:2]))
                    + r" \\\\ \vdots \\\\ "
                    + " \\\\ ".join(map(str, self[-2:]))
                    + r" \end{bmatrix}\right)"
                )
            else:
                latex_repr = (
                    r"Vector\left(\begin{bmatrix} "
                    + " \\\\ ".join(map(str, self))
                    + r" \end{bmatrix}\right)"
                )

            display(Math(latex_repr))
            return ""
        else:
            return f"Vector({', '.join(map(str, self))})"

    def add_vector(self, vector):
        validate_input(vector)

        base_vector = self

        # Ensure 'vector' is a Vector instance
        if not isinstance(vector, Vector):
            vector = Vector(vector)

        # Check if shapes are compatible
        if base_vector.shape != vector.shape:
            raise ValueError(
                f"Shapes must be equal, got {base_vector.shape} + {vector.shape}"
            )

        # Perform addition and return a new Vector instance
        return Vector(base_vector + vector)

    def subtract_vector(self, vector):
        validate_input(vector)
        base_vector = self

        # Ensure 'vector' is a Vector instance
        if not isinstance(vector, Vector):
            vector = Vector(vector)

        # Check if shapes are compatible
        if base_vector.shape != vector.shape:
            raise ValueError(
                f"Shapes must be equal, got {base_vector.shape} - {vector.shape}"
            )

        # Perform subtraction and return a new Vector instance
        return Vector(base_vector - vector)

    def scalar_multiply(self, scalar: int | float = 1.0):
        base_vector = self

        if not isinstance(scalar, (int, float)):
            raise TypeError(f"`scalar` must be of type `int|float`, got {type(scalar)}")

        return Vector(base_vector * scalar)


@decorator_validate_inputs
@decorator_validate_shapes
def is_dependent(*vectors) -> bool:
    if not vectors:
        raise ValueError("No input provided.")

    vectors = [Vector(v) if not isinstance(v, Vector) else v for v in vectors]
    matrix = np.column_stack(vectors)

    rank = np.linalg.matrix_rank(matrix)
    vector_num = matrix.shape[1]  # col num = vector num

    return bool(rank < vector_num)


@decorator_validate_shapes
def find_linear_combination(target_vector, *vectors) -> Optional[np.ndarray]:
    validate_input(target_vector)
    validate_input(vectors)

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


@decorator_validate_inputs
@decorator_validate_shapes
def find_span(*basis_vectors) -> dict:
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


@decorator_validate_inputs
@decorator_validate_shapes
def is_basis(*vectors) -> bool:
    return not is_dependent(*vectors)
