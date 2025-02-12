from .independent_utils import get_shape, can_be_vector
from typing import List
import math
from numbers import Number


class Vector:
    """
    A class used to represent a Vector.

    Attributes:
    ----------
    data : List[Number]
        A list containing the elements of the vector.

    Methods:
    -------
    Various methods to perform vector operations such as addition, subtraction, scaling,
    dot product, and more.

    Usage:
    -----
    This class is designed to perform vector operations without relying on external libraries
    like NumPy. It supports basic operations and some advanced linear algebra concepts.
    """

    def __init__(self, data: List[Number]):
        can_be_vector(data)
        self.data = data

    def __add__(self, other):
        if isinstance(other, Vector):
            if len(self.data) != len(other.data):
                raise ValueError("Both vectors must be of same length")
            return Vector([x + y for x, y in zip(self.data, other.data)])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector):
            if len(self.data) != len(other.data):
                raise ValueError("Both vectors must be of same length")
            return Vector([x - y for x, y in zip(self.data, other.data)])
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            scalar = other
            return Vector([x * scalar for x in self.data])
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError(
                    f"Shapes of vectors must match, got {self.shape} and {other.shape}"
                )
            return [x * y for x, y in zip(self.data, other.data)]
        raise TypeError(
            f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'"
        )

    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)
        raise TypeError(
            f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'Vector'"
        )

    def __str__(self):
        return f"Vector[ {"  ".join(str(component) for component in self.data)} ]"

    def __repr__(self):
        return f"Vector[ {"  ".join(str(component) for component in self.data)} ]"

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __neg__(self):
        return Vector([-x for x in self.data])

    def __matmul__(self, other):
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError(
                    f"Shapes of vectors must match, got {self.shape} and {other.shape}"
                )
            return sum([x * y for x, y in zip(self.data, other.data)])
        raise TypeError(
            f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'"
        )

    @property
    def magnitude(self):
        """
        Calculate the magnitude of the vector.

        Returns:
        -------
        float
            The magnitude of the vector.
        """
        return sum([x**2 for x in self.data]) ** 0.5

    @property
    def length(self):
        """
        Get the length (number of elements) of the vector.

        Returns:
        -------
        int
            The length of the vector.
        """
        return len(self.data)

    @property
    def shape(self):
        """
        Get the shape of the vector.

        Returns:
        -------
        tuple
            The shape of the vector.
        """
        return get_shape(self.data)

    @property
    def ndim(self):
        """
        Get the number of dimensions of the vector.

        Returns:
        -------
        int
            The number of dimensions of the vector.
        """
        return len(self.shape)

    def angle_between(self, other, in_degrees: bool = True):
        """
        Calculate the angle between this vector and another vector.

        Parameters:
        ----------
        other : Vector
            The other vector to calculate the angle with.
        in_degrees : bool, optional
            If True, return the angle in degrees. Otherwise, return the angle in radians.

        Returns:
        -------
        float
            The angle between the two vectors.
        """
        if not isinstance(other, Vector):
            raise TypeError("Argument should be of type Vector")
        if len(self.data) != len(other.data):
            raise ValueError("Both vectors must be of same length")

        # cos(theta) = (A o B) / (|A| * |B|)
        dot_product = self @ other
        magnitude_product = self.magnitude * other.magnitude
        cos_theta = dot_product / magnitude_product

        # theta = cos^-1(cos(theta))
        theta_radians = math.acos(cos_theta)
        if in_degrees:
            theta_degrees = theta_radians * (180 / math.pi)
            return theta_degrees
        return theta_radians

    # using numpy
    def cross_product(self, other):
        """
        Calculate the cross product of this vector and another vector.

        Parameters:
        ----------
        other : Vector
            The other vector to calculate the cross product with.

        Returns:
        -------
        Vector
            The cross product of the two vectors.

        Raises:
        ------
        ValueError
            If the vectors are not 3-dimensional.
        """
        can_be_vector(other)
        if not isinstance(other, Vector):
            raise TypeError("Argument should be of type Vector")
        if len(self.data) != 3 or len(other.data) != 3:
            raise ValueError("Both vectors must be 3-dimensional")

        x1, y1, z1 = self.data
        x2, y2, z2 = other.data

        cross_prod = [
            y1 * z2 - z1 * y2,
            z1 * x2 - x1 * z2,
            x1 * y2 - y1 * x2,
        ]

        return Vector(cross_prod)
