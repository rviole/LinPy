class Vector:
    def __init__(self, data):
        self.data = data
        self.shape = self.get_shape()

    def get_shape(self):
        return len(self.data)

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
        raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")

    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            scalar = other
            return Vector([x * scalar for x in self.data])
        raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'Vector'")
    def __str__(self):
        return f"Vector[{" ".join(str(component) for component in self.data)}]"


# Matrix from vectors + Matrix from 2d list
class Matrix:
    def __init__(self, data):
        pass


if __name__ == "__main__":
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    c = 5
    print(v1 + v2)
    print(v1 - v2)
    print(v1 * v2)
    print(c * v1)
    print(v1 * c)
    # print([1]*  v2)
    print(v1 *[1]) 
    print(v1.shape)



