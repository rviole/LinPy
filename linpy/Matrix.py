from .utils import get_shape, zeros, can_be_matrix, can_be_vector
from typing import List
from .Vector import Vector


# Matrix from vectors + Matrix from 2d list
class Matrix:
    def __init__(self, data: List[List[int | float]]):

        can_be_matrix(data)

        self.data = data
        self.shape = get_shape(self.data)
        self.ndim = len(self.shape)

    def __add__(self, other):
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Both matrices must be of same shape")

            new_matrix = zeros(self.shape)
            for i, row in enumerate(self.data):
                for j, element in enumerate(row):
                    new_matrix[i][j] = element + other.data[i][j]
            return Matrix(new_matrix)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Both matrices must be of same shape")

            new_matrix = zeros(self.shape)
            for i, row in enumerate(self.data):
                for j, element in enumerate(row):
                    new_matrix[i][j] = element - other.data[i][j]
            return Matrix(new_matrix)
        return NotImplemented

    def __mul__(self, other):

        # Matrix multiplication
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    f"Matrix multiplication not possible with shapes {self.shape} and {other.shape}. Need (m x n) and (n x p)"
                )
            # Finding a dot product of two matrices

            return self.apply_on_matrix(other)

        # Matrix-vector multiplication
        elif isinstance(other, Vector):
            return self.apply_on_vector(other)

        # Scalar multiplication
        elif isinstance(other, (int, float)):
            new_matrix = zeros(shape=self.shape)
            for i, row in enumerate(self.data):
                for j, element in enumerate(row):
                    new_matrix[i][j] = element * other
            return Matrix(new_matrix)

        raise TypeError(
            f"Unsupported operand type(s) for *: 'Matrix' and '{type(other).__name__}'"
        )

    def __rmul__(self, other):
        if isinstance(other, Vector):
            raise ValueError(
                "Must be `MATRIX x VECTOR` or `MATRIX x MATRIX` multiplication, not `VECTOR x MATRIX`"
            )
        if isinstance(other, (int, float)):
            return self.__mul__(other)

        raise TypeError(
            f"Unsupported operand type(s) for *: 'Matrix' and '{type(other).__name__}'"
        )

    def __str__(self):
        output = "Matrix["
        for row in self.data:
            output += "[ " + "  ".join([str(element) for element in row]) + " ]"
            output += "\n" + " " * len("Matrix[")
        output = output.strip() + "]"
        return output

    def __repr__(self):
        return f"Matrix[{self.data}]"

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @property
    def transpose(self):
        base_matrix = self.data
        new_matrix = zeros(shape=self.shape[::-1])
        for i, row in enumerate(base_matrix):
            for j, element in enumerate(row):
                new_matrix[j][i] = element
        return Matrix(new_matrix)

    @property
    def T(self):
        return self.transpose

    def apply_on_vector(self, vector):
        can_be_vector(vector)
        vector = Vector(vector)
        get_shape(vector)
        matrix = self.data

        # ensure shape compability (m, n) * (n,)
        if self.shape[1] != vector.shape[0]:
            raise Exception(
                f"Shapes must match (m, n) * (n,), got: {self.shape} * {vector.shape}"
            )

        # Usually each column of the matricx corresponds to a transformed basis vector
        # For calculation purposes, we will transpose the matrix, making each row a transformed basis vector
        matrix = self.T
        new_vector = []
        for i, element in enumerate(vector):
            basis_vector = matrix[i]
            val = [element * x for x in basis_vector]
            new_vector.append(val)
        # sum 2 vectors
        new_vector = [sum(components) for components in zip(*new_vector)]

        # Lets revise the logic
        # Here we apply the transformation matrix on the vector
        # Each column of the matrix is a transformed basis vector
        # So we multiply each element of the vector with the corresponding basis vector in the matrix
        # 1st component of the vector -> 1st column of the matrix
        # 2nd component of the vector -> 2nd column of the matrix
        # and so on
        # Why do we used transpose? Because we use python lists in calculations (and not numpy arrays), presenting vectors as rows is more convenient (they were columns in the matrix).

        return Vector(new_vector)

    def apply_on_matrix(self, matrix):
        # dot product of two matrices
        can_be_matrix(matrix)
        matrix = Matrix(matrix)

        # ensure shape compability (m, n) * (n, p)
        if self.shape[1] != matrix.shape[0]:
            raise Exception(
                f"Shapes must match (m, n) * (n, p), got: {self.shape} * {matrix.shape}"
            )

        # we will interpret the second matrix as a collection of vectors (each column is a vector)
        # for calculation purposes, we will transpose the matrix so each row is a vector
        matrix = matrix.T

        # Lets go over logic again
        # WE consider the first (base) matrix as a transformation matrix
        # and the second matrix as a collection of vectors (each column is a vector)
        # So we want to apply the transformation matrix on each vector
        # For this we just iterate over second matrix and apply the transformation matrix on each vector
        # Why do we use transpose on the second matrix? Because we use python lists in calculations (and not numpy arrays), presenting vectors as rows is more convenient (they were columns in the matrix).
        # THen why do we use transpose at the end of the function? Because vectors were appended as rows in the new matrix for calculation purposes, but we want to show them as columns.

        new_matrix = []
        for i, vector in enumerate(matrix):
            new_column = self.apply_on_vector(vector)
            new_matrix.append(new_column)

        # because columns are shown as rows in the matrix, we will transpose the matrix
        return Matrix(new_matrix).T
