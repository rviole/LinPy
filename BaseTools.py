import numpy as np
from IPython.display import display, Math


def validate_input(input_data, raise_exception=True) -> bool:
    if not input_data:
        if raise_exception:
            raise ValueError("No input provided.")
        return False
    return True


def validate_equal_shapes(*data, raise_exception=True) -> bool:
    if not data:
        raise ValueError("No input provided.")

    if all(each.shape == data[0].shape for each in data):
        return True
    else:
        if raise_exception:
            raise ValueError("Shapes must be equal.")
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
