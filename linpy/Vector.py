from .utils import get_shape, can_be_vector
from typing import List


class Vector:
    def __init__(self, data: List[int | float]):
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
        if isinstance(other, (int, float)):
            scalar = other
            return Vector([x * scalar for x in self.data])
        if isinstance(other, Vector):
            return Vector([x * y for x, y in zip(self.data, other.data)])
        raise TypeError(
            f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'"
        )

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
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

    @property
    def magnitude(self):
        # return magnitude of the vector
        return sum([x**2 for x in self.data]) ** 0.5
    
    @property
    def length(self):
        return len(self.data)
    
    @property
    def shape(self):
        return  get_shape(self.data)
    
    @property
    def ndim(self):
        return len(self.shape)
    