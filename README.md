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
- [x] Check if vector is a linear combination of basis vectors
- [x] Check if a set of vectors forms a basis
- [x] Verify linear independence of vectors inside matrix
- [X] Compute the span of a Matrix
- [x] Check if a matrix represents a linear transformation
- [x] Apply a linear transformation to a vector
- [x] Compose two linear transformations
- [x] Matrix multiplication
- [x] Calculate matrix inverse
- [x] Compute matrix determinant
- [x] Check if a matrix is invertible
<!-- - [ ] Compute column space of a matrix -->
<!-- - [ ] Compute row space of a matrix -->
- [X] Vector dot product
- [x] Matrix dot product
- [x] Matrix-vector dot product
- [x] Compute cross product of two vectors
- [x] Calculate angle between two vectors
<!-- - [ ] Compute null space of a matrix -->
- [ ] Transform a matrix to a new basis
- [x] Check if a matrix is diagonal
- [x] Check if a matrix is an identity matrix
- [x] Check if a matrix is anti-diagonal
- [x] Check if a matrix is upper triangular
- [x] Check if a matrix is lower triangular
- [x] Check if a matrix is symmetric
- [x] Check if a matrix is skew-symmetric
- [ ] Implement Transformation class

## **📖 Description**

LinPy is a Python package that demonstrates a deep understanding of linear algebra concepts through practical implementation. Unlike typical projects that rely on libraries like NumPy, LinPy is built using Python's built-in lists to perform vector and matrix operations, showcasing both theoretical knowledge and practical coding skills.

Limitations:
- only int|float components for Vector/Matrix instances are allowed.
- Vectors can only be 1D -> shape = (n,)
- Matrix can only be 2D -> shape = (n, m)

## **📚 Available Methods**

### **General Utilities**
| **Function**                   | **Description**                                      | **Parameters**                          | **Returns**                        |
|--------------------------------|------------------------------------------------------|-----------------------------------------|------------------------------------|
| `get_shape(data)`              | Get the shape of the data                            | `data: Iterable`                        | `Tuple[int]`                       |
| `zeros(shape)`                 | Create a zero matrix of given shape                  | `shape: Tuple[int, int]`                | `List[List[int]]`                  |
| `can_be_vector(data)`          | Validate if data can be a vector                     | `data: Iterable`                        | `bool`                             |
| `can_be_matrix(data)`          | Validate if data can be a matrix                     | `data: Iterable[Iterable]`              | `bool`                             |

### **Vector Methods**
| **Method**                     | **Description**                                      | **Parameters**                          | **Returns**                        |
|--------------------------------|------------------------------------------------------|-----------------------------------------|------------------------------------|
| `__init__(data)`               | Initialize a vector                                  | `data: List[Number]`                    | `None`                             |
| `angle_between(other, in_degrees)` | Calculate the angle between two vectors         | `other: Vector, in_degrees: bool`       | `float`                            |
| `magnitude`                    | Calculate the magnitude of the vector                | None                                    | `float`                            |
| `length`                       | Get the length of the vector                         | None                                    | `int`                              |
| `shape`                        | Get the shape of the vector                          | None                                    | `Tuple[int]`                       |
| `ndim`                         | Get the number of dimensions of the vector           | None                                    | `int`                              |
| `cross_product(other)`         | Calculate the cross product with another vector      | `other: Vector`                         | `Vector`                           |

### **Matrix Methods**
| **Method**                     | **Description**                                      | **Parameters**                          | **Returns**                        |
|--------------------------------|------------------------------------------------------|-----------------------------------------|------------------------------------|
| `__init__(data)`               | Initialize a matrix                                  | `data: List[List[Number]]`              | `None`                             |
| `apply_on_vector(vector)`      | Apply the matrix as a linear transformation on a vector | `vector: Vector`                     | `Vector`                           |
| `apply_on_matrix(matrix)`      | Apply the matrix as a linear transformation on another matrix | `matrix: Matrix`                 | `Matrix`                           |
| `compose(*matrices)`           | Compose the current matrix with one or more matrices | `*matrices: Matrix`                    | `Matrix`                           |
| `rank`                         | Calculate the rank of the matrix using NumPy         | None                                    | `int`                              |
| `is_full_rank`                 | Check if the matrix is of full rank                  | None                                    | `bool`                             |
| `determinant`                  | Calculate the determinant of the matrix using NumPy  | None                                    | `float`                            |
| `inverse`                      | Calculate the inverse of the matrix using NumPy      | None                                    | `Matrix`                           |
| `is_singular`                  | Check if the matrix is singular                      | None                                    | `bool`                             |
| `is_invertable`                | Check if the matrix is invertible                    | None                                    | `bool`                             |
| `is_linear_transformation`     | Check if the matrix represents a linear transformation | None                                  | `bool`                             |
| `is_linearly_dependent`        | Check if the columns of the matrix are linearly dependent | None                                | `bool`                             |
| `span`                         | Calculate the span of the matrix                     | None                                    | `dict`                             |
| `is_basis`                     | Check if the columns of the matrix form a basis      | None                                    | `bool`                             |
| `is_linear_combination(vector)`| Check if a vector is a linear combination of the columns of the matrix | `vector: Vector`              | `bool`                             |
