# LinPy
Portfolio Repository : A repository that showcases my knowledge in Linear Algebra.

Limitations:
- only int|float components for Vector/Matrix instances are allowed.
- Vectors can only be 1D -> shape = (n,)
- Matrix can only be 2D -> shape = (n, m)

# 🏗️In progress...🚧

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
- [ ] Matrix-vector dot product
- [ ] Compute cross product of two vectors
- [ ] Calculate angle between two vectors
- [ ] Compute null space of a matrix
- [ ] Transform a matrix to a new basis
- [ ] Check if a matrix is diagonal
- [ ] Check if a matrix is an identity matrix
- [ ] Check if a matrix is anti-diagonal
- [ ] Check if a matrix is upper triangular
- [ ] Check if a matrix is lower triangular
- [ ] Check if a matrix is symmetric
- [ ] Check if a matrix is skew-symmetric
- [ ] Implement Transformation class

## Structure
The structure of the project is following:
1. BaseTools.py - provide a basic functionality to create Vector() or Matrix() , etc . The same as np.array().
2. VectorTools.py - provide methods created specifically for vector operations, like is_basis() or calculate_linear_combination()
3. MatrixTools.py - provide methods created specifically for matrix operations, like is_linear_transformation(), or compose_transformation()

The Base Tools contain classes contain methods that use ready optimized solutions, whereas MAtrixTools and VectorTOols.py scripts use the same functions but from scratch. THis proves the deep understanding of both theorical and practical sides of the concept

