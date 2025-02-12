from .independent_utils import get_shape, zeros, can_be_matrix, can_be_vector
from typing import List
from .Vector import Vector
import numpy as np
from numbers import Number

# Matrix from vectors + Matrix from 2d list
class Matrix:
    """
    A class used to represent a Matrix.

    Attributes:
    ----------
    data : List[List[Number]]
        A 2D list containing the elements of the matrix.

    Methods:
    -------
    Various methods to perform matrix operations such as addition, subtraction, multiplication,
    transposition, inversion, and more.

    Usage:
    -----
    This class is designed to perform matrix operations without relying on external libraries
    like NumPy. It supports basic operations and some advanced linear algebra concepts.
    """

    def __init__(self, data: List[List[Number]]):

        can_be_matrix(data)
        self.data = data

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
        elif isinstance(other, Number):
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
        if isinstance(other, Number):
            return self.__mul__(other)

        raise TypeError(
            f"Unsupported operand type(s) for *: 'Matrix' and '{type(other).__name__}'"
        )

    def __matmul__(self, other):
        # Matrix multiplication
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    f"Matrix multiplication not possible with shapes {self.shape} and {other.shape}. Need (m x n) and (n x p)"
                )
            # Finding a dot product of two matrices

            return self.apply_on_matrix(other)

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

    def __neg__(self):
        new_matrix = zeros(shape=self.shape)
        for i, row in enumerate(self.data):
            for j, element in enumerate(row):
                new_matrix[i][j] = -element
        return Matrix(new_matrix)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError(f"Can't compare Matrix with {type(other).__name__}")
        return self.data == other.data

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

    @property
    def shape(self):
        return get_shape(self.data)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def diagonal(self):
        if not self.is_square:
            raise ValueError(
                f"Non-square matrix doesn't have a diagonal, got shape {self.shape}"
            )
        digonal_length = self.shape[0]
        diagonal_vector = zeros(shape=(digonal_length,))

        for i, row in enumerate(self.data):
            diagonal_vector[i] = self.data[i][i]

        return Vector(diagonal_vector)

    @property
    def anti_diagonal(self):
        if not self.is_square:
            raise ValueError(
                f"Non-square matrix doesn't have an anti-diagonal, got shape {self.shape}"
            )
        anti_digonal_length = self.shape[0]
        anti_diagonal_vector = zeros(shape=(anti_digonal_length,))

        for (
            i,
            row,
        ) in enumerate(self.data):
            anti_diagonal_vector[i] = self.data[i][anti_digonal_length - 1 - i]
        return Vector(anti_diagonal_vector)

    @property
    def trace(self):
        return sum(self.diagonal)

    @property
    def anti_trace(self):
        return sum(self.anti_diagonal)

    @property
    def is_symmetric(self):
        if not self.is_square:
            raise ValueError(
                f"Non-square matrix doesn't have a diagonal, got shape {self.shape}"
            )

        return self == self.T

    @property
    def is_square(self):
        if self.shape[0] == self.shape[1]:
            return True
        return False

    @property
    def is_diagonal(self):
        if not self.is_square:
            raise ValueError(
                f"Non-square matrix doesn't have a diagonal, got shape {self.shape}"
            )

        non_diagonal_values = []

        for i, row in enumerate(self.data):
            non_diagonal_values.extend(row[i + 1 :])
            non_diagonal_values.extend(row[:i])

        diagonal_validation = all(x != 0 for x in self.diagonal)
        non_diagonal_validation = all(x == 0 for x in non_diagonal_values)

        return diagonal_validation and non_diagonal_validation

    @property
    def is_anti_diagonal(self):
        if not self.is_square:
            raise ValueError(
                f"Non-square matrix doesn't have a diagonal, got shape {self.shape}"
            )
        non_anti_diagonal_values = []

        cols = self.shape[1]

        for i, row in enumerate(self.data):
            non_anti_diagonal_values.extend(row[: cols - i - 1])
            non_anti_diagonal_values.extend(row[cols - i :])

        anti_diagonal_validation = all(x != 0 for x in self.anti_diagonal)
        non_anti_diagonal_validation = all(x == 0 for x in non_anti_diagonal_values)

        return anti_diagonal_validation and non_anti_diagonal_validation

    @property
    def is_identity(self):
        if not self.is_square:
            raise ValueError(
                f"Non-square matrix doesn't have a diagonal, got shape {self.shape}"
            )

        non_diagonal_values = []

        for i, row in enumerate(self.data):
            non_diagonal_values.extend(row[i + 1 :])
            non_diagonal_values.extend(row[:i])

        diagonal_validation = all(x == 1 for x in self.diagonal)
        non_diagonal_validation = all(x == 0 for x in non_diagonal_values)

        return diagonal_validation and non_diagonal_validation

    @property
    def is_upper_triangular(self):
        if not self.is_square:
            raise ValueError(
                f"Non-square matrix doesn't have a diagonal, got shape {self.shape}"
            )

        over_diagonal = []
        under_diagonal = []

        for i, row in enumerate(self.data):
            over_diagonal.extend(row[i:])
            under_diagonal.extend(row[:i])

        # Check if all values over diagonal (including diagonal) are non-zero
        upper_validation = all([x != 0 for x in over_diagonal])

        # Check if all values under diagonal (excluding diagonal) are zero
        lower_validation = all([x == 0 for x in under_diagonal])

        return upper_validation and lower_validation

    @property
    def is_lower_triangular(self):
        if not self.is_square:
            raise ValueError(
                f"Non-square matrix doesn't have a diagonal, got shape {self.shape}"
            )

        over_diagonal = []
        under_diagonal = []

        for i, row in enumerate(self.data):
            over_diagonal.extend(row[i + 1 :])
            under_diagonal.extend(row[: i + 1])

        # Check if all values over diagonal (excluding diagonal) are non-zero
        upper_validation = all([x == 0 for x in over_diagonal])

        # Check if all values under diagonal (including diagonal) are zero
        lower_validation = all([x != 0 for x in under_diagonal])

        return upper_validation and lower_validation

    @property
    def is_skew_symmetric(self):
        if not self.is_square:
            raise ValueError(
                f"Non-square matrix doesn't have a diagonal, got shape {self.shape}"
            )

        return self == -self.T

    def apply_on_vector(self, vector):
        """
        Apply the matrix as a linear transformation on a vector.

        Parameters:
        ----------
        vector : Vector
            The vector to be transformed.

        Returns:
        -------
        Vector
            The transformed vector.
        """
        can_be_vector(vector)
        if not isinstance(vector, Vector):
            vector = Vector(vector)

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
        """
        Apply the matrix as a linear transformation on another matrix.

        Parameters:
        ----------
        matrix : Matrix
            The matrix to be transformed.

        Returns:
        -------
        Matrix
            The transformed matrix.
        """
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

    def compose(self, *matrices):
        """
        Compose the current matrix with one or more matrices.

        Parameters:
        ----------
        *matrices : Matrix
            One or more matrices to compose with the current matrix.

        Returns:
        -------
        Matrix
            The resulting matrix after composition.
        """
        # m1.compose(m2, m3, m4) = m1 @ m2 @ m3 @ m4
        # f.compose(g, h ) = f(g(h(x)))

        if not all([isinstance(matrix, Matrix) for matrix in matrices]):
            raise ValueError("All arguments must be of type Matrix")
        if not all([self.shape[1] == matrix.shape[0] for matrix in matrices]):
            raise ValueError("Shapes of matrices are not compatible for composition")

        # Compose matrices
        result = self
        for matrix in matrices:
            result = result @ matrix
        return result

    # using numpy

    @property
    def rank(self):
        """
        Calculate the rank of the matrix using NumPy.

        Returns:
        -------
        int
            The rank of the matrix.
        """
        return np.linalg.matrix_rank(self.data)

    @property
    def is_full_rank(self):
        """
        Check if the matrix is of full rank.

        Returns:
        -------
        bool
            True if the matrix is of full rank, False otherwise.
        """
        n_cols = self.shape[1]
        return n_cols == self.rank

    @property
    def determinant(self):
        """
        Calculate the determinant of the matrix using NumPy.

        Returns:
        -------
        float
            The determinant of the matrix.
        """
        return np.linalg.det(self.data)

    @property
    def inverse(self):
        """
        Calculate the inverse of the matrix using NumPy.

        Returns:
        -------
        Matrix
            The inverse of the matrix.

        Raises:
        ------
        ValueError
            If the matrix is not square or not of full rank.
        """
        if not self.is_square:
            raise ValueError("Inverse can only be calculated for square matrices")
        if not self.is_full_rank:
            raise ValueError("Inverse can only be calculated for full rank matrices")
        try:
            return Matrix(np.linalg.inv(self.data))
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular, can't calculate inverse")

    @property
    def is_singular(self):
        """
        Check if the matrix is singular (not of full rank).

        Returns:
        -------
        bool
            True if the matrix is singular, False otherwise.
        """
        return not self.is_full_rank

    @property
    def is_invertable(self):
        """
        Check if the matrix is invertible.

        Returns:
        -------
        bool
            True if the matrix is invertible, False otherwise.
        """
        try:
            self.inverse
            return True
        except ValueError:
            return False

    @property
    def is_linear_transformation(self):
        """
        Check if the matrix represents a linear transformation.

        Returns:
        -------
        bool
            True if the matrix represents a linear transformation, False otherwise.
        """
        return self.is_full_rank

    @property
    def is_linearly_dependent(self):
        """
        Check if the columns of the matrix are linearly dependent.

        Returns:
        -------
        bool
            True if the columns are linearly dependent, False otherwise.
        """
        n_cols = self.shape[1]
        return not self.rank == n_cols

    @property
    def span(self):
        """
        Calculate the span of the matrix.

        Returns:
        -------
        dict
            A dictionary containing the rank, full_span, and zero_span of the matrix.
        """
        n_cols = self.shape[1]

        output = {
            "rank": int(self.rank),  # Dimension of the span (rank)
            "full_span": False,  # Whether the span fills the entire space
            "zero_span": False,  # Whether the span is trivial (zero vector only)
        }

        if self.rank == 0:
            output["zero_span"] = True
        elif self.is_full_rank:
            output["full_span"] = True

        return output

    @property
    def is_basis(self):
        """
        Check if the columns of the matrix form a basis.

        Returns:
        -------
        bool
            True if the columns form a basis, False otherwise.
        """
        return not self.is_linearly_dependent

    def is_linear_combination(self, vector: Vector):
        """
        Check if a vector is a linear combination of the columns of the matrix.

        Parameters:
        ----------
        vector : Vector
            The vector to check.

        Returns:
        -------
        bool
            True if the vector is a linear combination of the columns, False otherwise.
        """
        if not isinstance(vector, Vector):
            raise ValueError(f"Expected a Vector, got {type(vector).__name__}")
        if self.shape[0] != vector.shape[0]:
            raise ValueError(
                f"Matrix and vector must have same number of rows, got {self.shape[0]} and {vector.shape[0]}"
            )
        new_matrix = [v for v in self.T]
        new_matrix.append(vector.data)
        print(new_matrix)
        new_matrix = Matrix(new_matrix)

        return new_matrix.rank == self.rank
