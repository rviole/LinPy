from typing import Tuple, List


def get_shape(data) -> Tuple[int | float]:

    def can_find_length(data):
        try:
            len(data)
            return True
        except Exception as e:
            return False

    def validate_same_shapes(*data, raise_exception=True):
        shapes = [len(x) for x in data]
        if len(set(shapes)) != 1:
            if raise_exception:
                raise ValueError(
                    f"Not all shapes are same. Shapes for each element {shapes}"
                )
            return False
        return True

    def validate_same_types(*data, raise_exception=True):
        types = (type(x) for x in data)
        if len(set(types)) != 1:
            if raise_exception:
                raise TypeError(f"Not all data types in the data are same")
            return False
        return True

    def _get_shape(idx, n_rows, n_cols, ndims, *data):
        inner_data = data[0]
        if not can_find_length(inner_data):
            validate_same_types(*data, raise_exception=True)
            if ndims == 1:
                return (n_cols,)  # for 1D
            elif ndims == 0:  # for scalar
                return (0,)
            return n_rows, n_cols
        else:
            idx += 1
            ndims += 1
            if idx > 2:
                raise Exception(
                    f"The passed array has {ndims} dimensions, allowed dimensions are 1D and 2D."
                )

            validate_same_types(*data, raise_exception=True)
            validate_same_shapes(*data, raise_exception=True)
            n_cols = len(inner_data)
            n_rows = len(data)
            return _get_shape(idx, n_rows, n_cols, ndims, *inner_data)

    shape = _get_shape(0, 0, 0, 0, data)
    return shape


class Vector:
    def __init__(self, data: List[int | float]):
        self.data = data
        self.shape = get_shape(self.data)
        self.ndim = len(self.shape)

        ndim_validiton = self.ndim == 1
        type_validation = all(isinstance(x, (int, float)) for x in data)

        if not ndim_validiton:
            raise ValueError(f"A Vector must be 1D, got {self.ndim}D instead.")
        if not type_validation:
            raise TypeError("Vector can only contain integers or floats")

        self.length = len(self.data)

    def __add__(self, other):
        return Vector([x + y for x, y in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([x - y for x, y in zip(self.data, other.data)])

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
            scalar = other
            return Vector([x * scalar for x in self.data])
        raise TypeError(
            f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'Vector'"
        )

    def __str__(self):
        return f"Vector[{" ".join(str(component) for component in self.data)}]"

    def __len__(self):
        return len(self.data)


# Matrix from vectors + Matrix from 2d list
class Matrix:
    def __init__(self, data: List[List[int | float]]):
        self.data = data
        self.shape = get_shape(self.data)
        self.ndim = len(self.shape)

        ndim_validation = self.ndim == 2
        type_validation = all(
            isinstance(element, (int, float)) for row in self.data for element in row
        )
        if not ndim_validation:
            raise ValueError(f"A Matrix must be 2D, got {self.ndim}D instead.")
        if not type_validation:
            raise TypeError("Matrix can only contain integers or floats")

    
if __name__ != "__main__":
    v1 = Vector([1, 2, 3, 12])
    v2 = Vector([4, 5, 6])
    c = 5
    print(v1 + v2)
    print(v1 - v2)
    print(v1 * v2)
    print(c * v1)
    print(v1 * c)
    print(len(v1))
    print(type(v1))
    # print([1]*  v2)
    # print(v1 *[1])
    print(v1.shape)


list_1D = [1, 2, 3]
list_2D = [[1, 2], [4, 5], [7, 8]]
lsit_2D2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
list_3D = [[[1], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]

# get_shape(list_1D)
# get_shape(list_2D)
# get_shape(lsit_2D2)
# print(get_shape(list_2D))


Matrix([[1, 2], [3, 4.5]])
