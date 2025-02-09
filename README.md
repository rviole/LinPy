# LinPy
A repository that showcases my knowledge in Linear Algebra.

<br>

## **🔍 Portfolio Project Showcase**  
*This is a conceptual demonstration, not production-ready code.*  

### **✨ Key Concepts Demonstrated**  
| **Concept**               | **Implementation Example**                      |  
|---------------------------|-------------------------------------------------|  
| **Vector Operations**     | Custom vector addition, subtraction, and scaling |  
| **Matrix Operations**     | Matrix multiplication, transposition, and inversion |  
| **Linear Transformations**| Applying and composing linear transformations |  

<br>

**Key Notes:**  
⚠️ Prioritizes mathematical clarity over practicality  
⚠️ Intentional unoptimized implementation **to showcase step-by-step logic and enhance code clarity**  
⚠️ Serves as skills exhibit for technical interviews  

<br>

## **🚧 Still in Progress**  

## To be Implemented
- [x] Vector-vector subtraction
- [x] Vector-scalar multiplication
- [x] Vector-vector addition
- [x] Matrix transposition
- [x] Find matrix diagonal
- [x] Find matrix anti-diagonal
- [x] Calculate matrix trace
- [x] Calculate matrix anti-trace
- [ ] Check if vector is a linear combination of basis vectors
- [ ] Check if a set of vectors forms a basis
- [ ] Verify linear independence of vectors
- [ ] Compute the span of a set of vectors
- [ ] Check if a matrix represents a linear transformation
- [ ] Apply a linear transformation to a vector
- [ ] Compose two linear transformations
- [x] Matrix multiplication
- [ ] Calculate matrix inverse
- [ ] Compute matrix determinant
- [ ] Check if a matrix is invertible
- [ ] Compute column space of a matrix
- [ ] Compute row space of a matrix
- [X] Vector dot product
- [x] Matrix dot product
- [x] Matrix-vector dot product
- [ ] Compute cross product of two vectors
- [x] Calculate angle between two vectors
- [ ] Compute null space of a matrix
- [ ] Transform a matrix to a new basis
- [x] Check if a matrix is diagonal
- [x] Check if a matrix is an identity matrix
- [ ] Check if a matrix is anti-diagonal
- [x] Check if a matrix is upper triangular
- [x] Check if a matrix is lower triangular
- [x] Check if a matrix is symmetric
- [x] Check if a matrix is skew-symmetric
- [ ] Implement Transformation class




## **📖 Description**

LinPy is a Python package that demonstrates a deep understanding of linear algebra concepts through practical implementation. Unlike typical projects that rely on libraries like NumPy, LinPy is built using Python's built-in lists to perform vector and matrix operations, showcasing both theoretical knowledge and practical coding skills.

## **📂 Project Structure**  
| File                     | Purpose                                  |  
|--------------------------|------------------------------------------|  
| `__init__.py`            | Package initialization and exports       |  
| `utils.py`               | Utility functions for shape validation and transformations |  
| `Vector.py`              | Defines the Vector class and its operations |  
| `Matrix.py`              | Defines the Matrix class and its operations |  

Limitations:
- only int|float components for Vector/Matrix instances are allowed.
- Vectors can only be 1D -> shape = (n,)
- Matrix can only be 2D -> shape = (n, m)

<!-- ---
## Personal Notes
This is the list of functions the repo will probably have:

### Vector Operations
1. `add_vectors(v1, v2)`
2. `subtract_vector`
3. `scalar_multiply(vector, scalar)`
4. `is_linear_combination(vector, basis_vectors)`

### Vector Spaces
5. `find_span(basis_vectors)`
6. `is_basis(vectors)`
7. `is_linearly_independent(vectors)`

### Linear Transformations
8. `is_linear_transformation(matrix)`
9. `apply_transformation(matrix, vector)`
10. `compose_transformations(matrix1, matrix2)`

### Matrices
11. `matrix_multiply(matrix1, matrix2)`
12. `transpose(matrix)`
13. `inverse_matrix(matrix)`

### Determinants and Properties
14. `calculate_determinant(matrix)`
15. `is_invertible(matrix)`
16. `calculate_trace`
17. `calculate_diagonal_sum`
18. `calculate_antidiagonal_sum`
19. `find_diagonal`
20. `find_anti_diagonal`

### Utility Classes
21. `Vector`
22. `Matrix`
23. `Transformation`

### Advanced Concepts
24. `calculate_column_space(matrix)`
25. `calculate_row_space`

### Dot and Cross Products
26. `vector_dot`
27. `matrix_dot`
28. `matrix_vector_dot`
29. `cross_product(v1, v2)`
30. `angle_between_vectors(v1, v2)`

31. `null_space(matrix)`
32. `change_of_basis(matrix, new_basis)`
 -->
