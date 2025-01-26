import unittest
import numpy as np
from VectorTools import Vector, Matrix, is_dependent


# Vectors


class TestVector(unittest.TestCase):

    def test_valid_vector(self):
        v = Vector([1, 2, 3])
        self.assertTrue(np.array_equal(v.data, np.array([1, 2, 3])))

    def test_2d_column_vector(self):
        v = Vector([[1], [2], [3]])
        self.assertTrue(np.array_equal(v.data, np.array([1, 2, 3])))

    def test_invalid_vector(self):
        with self.assertRaises(ValueError):
            Vector([[1, 2], [3, 4]])  # Not a valid 1D array


# Matrix


class TestMatrix(unittest.TestCase):

    def test_valid_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertTrue(np.array_equal(m.data, np.array([[1, 2], [3, 4]])))

    def test_1d_array_to_matrix(self):
        m = Matrix([1, 2, 3])
        self.assertTrue(np.array_equal(m.data, np.array([[1, 2, 3]])))

    def test_column_matrix_to_row_vector(self):
        m = Matrix([[1], [2], [3]])
        self.assertTrue(np.array_equal(m.data, np.array([[1, 2, 3]])))

    def test_invalid_matrix(self):
        with self.assertRaises(ValueError):
            Matrix([[[1, 2]]])  # Not a valid 1D or 2D array

    def test_rank(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.rank(), 2)


# Independent / Dependent Vectors


class TestIsDependent(unittest.TestCase):

    def test_independent_vectors(self):
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        v3 = [0, 0, 1]
        self.assertFalse(is_dependent(v1, v2, v3))

    def test_dependent_vectors(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        v3 = [5, 7, 9]  # v3 = v1 + v2
        self.assertTrue(is_dependent(v1, v2, v3))

    def test_single_matrix(self):
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertTrue(is_dependent(m))

    def test_independent_matrix(self):
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertFalse(is_dependent(m))

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            is_dependent()


if __name__ == "__main__":
    unittest.main()
