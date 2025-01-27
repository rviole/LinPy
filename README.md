# LinearAlgebra-portfolio
Portfolio Repository : A repository that showcases my knowledge in Linear Algebra.

# 🏗️In progress...🚧


## Structure
The structure of the project is following:
1. BaseTools.py - provide a basic functionality to create Vector() or Matrix() , etc . The same as np.array().
2. VectorTools.py - provide methods created specifically for vector operations, like is_basis() or calculate_linear_combination()
3. MatrixTools.py - provide methods created specifically for matrix operations, like is_linear_transformation(), or compose_transformation()



This is the list of functions the repo will probably have:

### Vector Operations
1. **`add_vectors(v1, v2)`**: Perform vector addition.
2. **`scalar_multiply(vector, scalar)`**: Multiply a vector by a scalar.
3. **`is_linear_combination(vector, basis_vectors)`**: Check if a vector can be expressed as a linear combination of given basis vectors.

### Vector Spaces
4. **`find_span(basis_vectors)`**: Compute the span of a set of vectors.
5. **`is_basis(vectors)`**: Check if a set of vectors forms a basis.
6. **`is_linearly_independent(vectors)`**: Verify linear independence of vectors.

### Linear Transformations
7. **`is_linear_transformation(matrix)`**: Check if a given transformation matrix represents a linear transformation.
8. **`apply_transformation(matrix, vector)`**: Apply a linear transformation to a vector.
9. **`compose_transformations(matrix1, matrix2)`**: Compose two linear transformations.

### Matrices
10. **`matrix_multiply(matrix1, matrix2)`**: Multiply two matrices.
11. **`transpose(matrix)`**: Find the transpose of a matrix.
12. **`inverse_matrix(matrix)`**: Calculate the inverse of a matrix if it exists.

### Determinants and Properties
13. **`calculate_determinant(matrix)`**: Compute the determinant of a square matrix.
14. **`is_invertible(matrix)`**: Check if a matrix is invertible.

### Advanced Concepts
15. **`null_space(matrix)`**: Compute the null space of a matrix.
16. **`column_space(matrix)`**: Compute the column space of a matrix.
17. **`change_of_basis(matrix, new_basis)`**: Transform a matrix to a new basis.

### Dot and Cross Products
18. **`dot_product(v1, v2)`**: Compute the dot product of two vectors.
19. **`cross_product(v1, v2)`**: Compute the cross product of two 3D vectors.
20. **`angle_between_vectors(v1, v2)`**: Calculate the angle between two vectors using their dot product.

### Utility Classes
21. **`Vector`**: A class to represent vectors with methods for addition, scaling, and dot products.
22. **`Matrix`**: A class to represent matrices with methods for multiplication, transposition, and inversion.
23. **`Transformation`**: A class to represent and apply linear transformations.

This list is extensive enough to cover both basic and advanced Linear Algebra concepts.