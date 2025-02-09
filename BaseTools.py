import numpy as np
from IPython.display import display, Math

np.random.seed(42)


# checks if any of the input data is None
def validate_input(*input_data, raise_exception=True) -> bool:
    if any(each is None for each in input_data):
        if raise_exception:
            raise ValueError("No input provided.")
        return False
    return True


# note her that data must be an iterable of numpy array-like objects
def validate_equal_shapes(*data, raise_exception=True) -> bool:
    validate_input(data)

    if all(each.shape == data[0].shape for each in data):
        return True
    else:
        if raise_exception:
            raise ValueError("Shapes must be equal.")
        return False


# note her that data must be an iterable of numpy array-like objects
def validate_multiplication_compatibility(obj_1, obj_2, raise_exception=True) -> bool:
    validate_input(obj_1, obj_2)

    # check if the number of columns in the matrix is equal to the number of rows in the vector
    if obj_1.shape[1] == obj_2.shape[0]:
        return True
    else:
        if raise_exception:
            raise ValueError(
                f"Two objects are not compatible. Got {obj_1.shape} - {obj_2.shape}. Shapes must be (m, n) - (n, p)."
            )
        return False


def validate_types(*data, cls, raise_exception=True) -> bool:
    validate_input(data, cls)

    if all(isinstance(each, cls) for each in data):
        return True
    else:
        if raise_exception:
            raise TypeError(f"Data types must be {cls}.")
        return False


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

        validate_input(data)
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

    def __str__(self):
        return f"Vector({', '.join(map(str, self))})"

    def add_vector(self, vector):

        validate_input(vector)

        base_vector = self
        validate_equal_shapes(base_vector, vector)

        # Ensure 'vector' is a Vector instance
        if not isinstance(vector, Vector):
            vector = Vector(vector)

        # Perform addition and return modified Vector
        return base_vector + vector

    def subtract_vector(self, vector):
        validate_input(vector)
        base_vector = self
        validate_equal_shapes(base_vector, vector)

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

    def vector_dot(self, vector):
        validate_input(vector)
        base_vector = self
        validate_equal_shapes(base_vector, vector)

        # Ensure 'vector' is a Vector instance
        if not isinstance(vector, Vector):
            vector = Vector(vector)

        # Perform dot product and return the result
        return np.dot(base_vector, vector)


class Matrix(np.ndarray):
    """
    A class representing a mathematical matrix that inherits from numpy.ndarray.

    This ensures that the Matrix behaves like a numpy array and supports all
    operations such as addition, multiplication, etc.

    Attributes:
        data (numpy.ndarray): The data of the matrix as a 2D numpy array.
    """

    def __new__(cls, *data, make_from_vectors=False):
        validate_input(data)
        # Case when creating from vectors
        if make_from_vectors:
            # Ensure multiple vectors are passed
            if len(data) < 2:
                raise ValueError("At least two vectors are required to form a matrix.")
            vectors = [Vector(v) if not isinstance(v, Vector) else v for v in data]
            validate_equal_shapes(*vectors)
            obj = np.column_stack(vectors)
        else:
            # Case when creating from a single matrix
            if len(data) != 1:
                raise ValueError(
                    "Only one argument is allowed when creating a matrix. To create a matrix from vectors, use `make_from_vectors=True`."
                )
            obj = np.asarray(data[0])

            if obj.ndim != 2:
                raise ValueError(f"A matrix must be a 2D array, got {obj.ndim}D.")

        return obj.view(cls)

    def rank(self):
        base_matrix = self
        return np.linalg.matrix_rank(base_matrix)

    def is_square(self):
        base_matrix = self
        return base_matrix.shape[0] == base_matrix.shape[1]

    def apply_on_vector(self, vector):
        validate_input(vector)

        if not isinstance(vector, Vector):
            vector = Vector(vector)

        base_matrix = self
        validate_multiplication_compatibility(base_matrix, vector)

        return base_matrix @ vector

    def get_transpose(self):
        base_matrix = self
        return base_matrix.T

    def get_inverse(self):
        base_matrix = self
        if not base_matrix.is_square():
            raise ValueError("Only square matrices can be inverted.")
        try:
            inv_matrix = np.linalg.inv(base_matrix)
        except np.linalg.LinAlgError as e:
            raise ValueError("Matrix is singular and cannot be inverted.")
        return Matrix(inv_matrix)

    def maltiply_matrix(self, matrix):
        validate_input(matrix)
        base_matrix = self
        if not isinstance(matrix, Matrix):
            matrix = Matrix(matrix)
        validate_multiplication_compatibility(base_matrix, matrix)
        return base_matrix @ matrix

    def get_trace(self):
        base_matrix = self
        if not base_matrix.is_square():
            raise ValueError("Only square matrices have a trace.")
        return np.trace(base_matrix)

    def get_determinant(self):
        base_matrix = self
        if not base_matrix.is_square():
            raise ValueError("Only square matrices have a determinant.")
        determinant = np.linalg.det(base_matrix)
        return determinant.astype(np.float64)

    def get_diagonal_sum(self):
        return self.get_trace()

    def get_antidigonal_sum(self):
        base_matrix = self
        if not base_matrix.is_square():
            raise ValueError("Only square matrices have an anti-diagonal.")
        return np.trace(np.fliplr(base_matrix))

    def get_diagonal(self):
        base_matrix = self
        if not base_matrix.is_square():
            raise ValueError("Only square matrices have a diagonal.")
        return np.diag(base_matrix)

    def get_anti_diagonal(self):
        base_matrix = self
        if not base_matrix.is_square():
            raise ValueError("Only square matrices have an anti-diagonal.")
        return np.diag(np.fliplr(base_matrix))

    def get_column_space(self):
        base_matrix = self
        rank = base_matrix.rank()
        column_space = base_matrix[:, :rank]
        return column_space

    def get_row_space(self):
        base_matrix = self
        row_space = self.get_column_space(base_matrix.get_transpose())
        return row_space

    def matrix_dot(self, matrix):
        validate_input(matrix)
        base_matrix = self
        if not isinstance(matrix, Matrix):
            matrix = Matrix(matrix)
        validate_multiplication_compatibility(base_matrix, matrix)

        return base_matrix @ matrix

    def vector_dot(self, vector):
        validate_input(vector)
        base_matrix = self
        if not isinstance(vector, Vector):
            vector = Vector(vector)
        validate_multiplication_compatibility(base_matrix, vector)

        return base_matrix @ vector


def matrix_vector_dot(matrix, vector):
    validate_input(matrix, vector)
    matrix = Matrix(matrix) if not isinstance(matrix, Matrix) else matrix
    vector = Vector(vector) if not isinstance(vector, Vector) else vector

    validate_multiplication_compatibility(matrix, vector)

    n_components = len(vector)
    result = Vector(
        np.zeros(
            n_components,
        )
    )

    for i in range(n_components):
        # each component of vector multiplied by corresponding columnn of the matrix
        result += vector[i] * matrix[:, i]
    return result


if __name__ == "__main__":
    # Test cases for Matrix class

    # Test creating a matrix from a 2D array
    try:
        matrix1 = Matrix([[1, 2], [3, 4]])
        print("Matrix1 created successfully:", matrix1)
    except Exception as e:
        print("Failed to create Matrix1:", e)

    # Test creating a matrix from vectors
    try:
        vector1 = Vector([1, 2])
        vector2 = Vector([3, 4])
        matrix2 = Matrix(vector1, vector2, make_from_vectors=True)
        print("Matrix2 created successfully from vectors:\n", matrix2)
    except Exception as e:
        print("Failed to create Matrix2 from vectors:", e)

    # Test rank method
    try:
        rank = matrix1.rank()
        print("Rank of Matrix1:", rank)
    except Exception as e:
        print("Failed to compute rank of Matrix1:\n", e)

    # Test is_square method
    try:
        is_square = matrix1.is_square()
        print("Matrix1 is square:", is_square)
    except Exception as e:
        print("Failed to check if Matrix1 is square:", e)

    # Test apply_on_vector method
    try:
        result_vector = matrix1.apply_on_vector(vector1)
        print("Result of applying Matrix1 on Vector1:", result_vector)
    except Exception as e:
        print("Failed to apply Matrix1 on Vector1:", e)

    # Test invalid matrix creation
    try:
        invalid_matrix = Matrix([1, 2, 3])
        print("Invalid matrix created successfully:", invalid_matrix)
    except Exception as e:
        print("Failed to create invalid matrix:", e)

    # Test invalid matrix creation from vectors with different shapes
    try:
        vector3 = Vector([1, 2, 3])
        invalid_matrix2 = Matrix(vector1, vector3, make_from_vectors=True)
        print("Invalid matrix2 created successfully from vectors:", invalid_matrix2)
    except Exception as e:
        print("Failed to create invalid matrix2 from vectors:", e)
