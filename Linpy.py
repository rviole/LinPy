def get_shape(data):

    def can_find_length(data):
        try:
            len(data)
            return True
        except Exception as e:
            return False
        
    def validate_equal_shapes(*data, raise_exception=True):
        shapes = [len(x) for x in data]
        if len(set(shapes)) != 1:
            if raise_exception:
                raise ValueError(f"Not all shapes are equal: {shapes}")
            return False
        return True


    def _get_shape(n_rows, n_cols, dim, *data):
        inner_data = data[0]
        if not can_find_length(inner_data):
            if dim == 1:
                return (n_cols,)  # for 1D
            elif dim == 0:  # for scalar
                return (0,)
            return n_rows, n_cols
        else:
            validate_equal_shapes(*data, raise_exception=True)
            n_cols = len(inner_data)
            n_rows = len(data)
            dim += 1
            return _get_shape(n_rows, n_cols, dim, *inner_data)

    shape = _get_shape(0, 0, 0, data)
    return shape





class Vector:
    def __init__(self, data):
        self.data = data
        self.shape = get_shape(self.data)
        self.ndim = len(self.shape)
        
        
        # Check if all components are numbers (of the same type)
        if not all(isinstance(x, (int, float)) for x in self.data):
            raise ValueError("All components must be numbers of the same type [int|float]")
        
        if self.ndim != 1:
            raise ValueError(f"Vector must be 1D, got {self.ndim}D")
        
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
    def __init__(self, data):
        pass


if __name__ == "__main__":
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


def is_shape_1D(data):
    return isinstance(data, list) and all(isinstance(x, (int, float)) for x in data)


print(get_shape(list_2D) == (3,2))
print(get_shape(lsit_2D2) == (3,3))
print(get_shape(list_1D) == (3,))
print(get_shape(1) == (0,))
