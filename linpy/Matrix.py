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
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    f"Matrix multiplication not possible with shapes {self.shape} and {other.shape}. Need (m x n) and (n x p)"
                )
            # Finding a dot product of two matrices
            new_matrix = zeros((self.shape[0], other.shape[1]))

            pass

        # finish the dot product

        return NotImplemented

    def __str__(self):
        return f'Matrix[{"\n       ".join([f"{row}".replace(",", "") for row in self.data])}]'

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    @property
    def transpose(self):
        base_matrix = self.data
        new_matrix = zeros(shape=self.shape)
        for i, row in enumerate(base_matrix):
            for j, element in enumerate(row):
                new_matrix[j][i] = element
        return Matrix(new_matrix)

    @property
    def T(self):
        return self.transpose

    def apply_on_vector(self, vector):
        print(vector)
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
        new_vector  = []
        for i, element in enumerate(vector):
            basis_vector = matrix[i]
            val = [element * x for x in basis_vector]
            print(val)
            new_vector.append(val)
        # sum 2 vectors
        new_vector = [sum(components) for components in zip(*new_vector)]
        print("sum:", Vector(new_vector))
