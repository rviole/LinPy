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

## **📖 Description**

LinPy is a Python package that demonstrates a deep understanding of linear algebra concepts through practical implementation. Unlike typical projects that rely on libraries like NumPy, LinPy is built using Python's built-in lists to perform vector and matrix operations, showcasing both theoretical knowledge and practical coding skills.

Limitations:
- only int|float components for Vector/Matrix instances are allowed.
- Vectors can only be 1D -> shape = (n,)
- Matrix can only be 2D -> shape = (n, m)








<!-- Edit the table properties columns -->
<!-- Test the code and imporve -->






## **📦 Usage**

### Installation
Clone the repository from GitHub:
```sh
git clone https://github.com/yourusername/LinPy.git
cd LinPy
```

### Importing
Import the package:
```python
import linpy as lp
```

### Creating Vectors and Matrices
Create a few vectors and matrices:
```python
import linpy as lp

# Create vectors
v1 = lp.Vector([1, 2, 3])
v2 = lp.Vector([4, 5, 6])

# Create matrices
m1 = lp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
m2 = lp.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

```

Using Numpy arrays:

```python
import numpy as np

# Create vectors from NumPy arrays
arr = np.array([1, 2, 3])
v3 = lp.Vector(arr)

# Create matrices from NumPy arrays
arr2 = np.array([[1, 2], [3, 4]])
m3 = lp.Matrix(arr2)
```

### Using Functions and Properties

#### Vector Operations
```python
print("v1 + v2 =", v1 + v2)
print("v1 - v2 =", v1 - v2)
print("v1 * 2 =", v1 * 2)
print("v1 . v2 =", v1 @ v2)
print("v1 x v2 =", v1.cross_product(v2))
print("Angle between v1 and v2 (degrees) =", v1.angle_between(v2))
```

#### Vector Properties
```python
print("Magnitude of v1 =", v1.magnitude)
print("Length of v1 =", v1.length)
print("Shape of v1 =", v1.shape)
print("Number of dimensions of v1 =", v1.ndim)
```

#### Matrix Operations
```python
print("m1 + m2 =", m1 + m2)
print("m1 - m2 =", m1 - m2)
print("m1 * 2 =", m1 * 2)
print("m1 * v1 =", m1.apply_on_vector(v1))
print("m1 * m2 =", m1.apply_on_matrix(m2))
print("m1 composed with m2 =", m1.compose(m2))
```

#### Matrix Properties
```python
print("Transpose of m1 =", m1.transpose)
print("Diagonal of m1 =", m1.diagonal)
print("Anti-diagonal of m1 =", m1.anti_diagonal)
print("Trace of m1 =", m1.trace)
print("Anti-trace of m1 =", m1.anti_trace)
print("Is m1 symmetric? =", m1.is_symmetric)
print("Is m1 square? =", m1.is_square)
print("Is m1 diagonal? =", m1.is_diagonal)
print("Is m1 anti-diagonal? =", m1.is_anti_diagonal)
print("Is m1 an identity matrix? =", m1.is_identity)
print("Is m1 upper triangular? =", m1.is_upper_triangular)
print("Is m1 lower triangular? =", m1.is_lower_triangular)
print("Is m1 skew-symmetric? =", m1.is_skew_symmetric)
print("Rank of m1 =", m1.rank)
print("Is m1 full rank? =", m1.is_full_rank)
print("Determinant of m1 =", m1.determinant)
print("Inverse of m1 =", m1.inverse)
print("Is m1 singular? =", m1.is_singular)
print("Is m1 invertible? =", m1.is_invertable)
print("Is m1 a linear transformation? =", m1.is_linear_transformation)
print("Are columns of m1 linearly dependent? =", m1.is_linearly_dependent)
print("Span of m1 =", m1.span)
print("Do columns of m1 form a basis? =", m1.is_basis)
print("Is v1 a linear combination of columns of m1? =", m1.is_linear_combination(v1))
```

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
| `cross_product(other)`         | Calculate the cross product with another vector      | `other: Vector`                         | `Vector`                           |


### **Vector Properties**
| **Property**                   | **Description**                                      |
|--------------------------------|------------------------------------------------------|
| `magnitude`                    | Calculate the magnitude of the vector                |
| `length`                       | Get the length of the vector                         |
| `shape`                        | Get the shape of the vector                          |
| `ndim`                         | Get the number of dimensions of the vector           |








### **Matrix Methods**
| **Method**                     | **Description**                                      | **Parameters**                          | **Returns**                        |
|--------------------------------|------------------------------------------------------|-----------------------------------------|------------------------------------|
| `__init__(data)`               | Initialize a matrix                                  | `data: List[List[Number]]`              | `None`                             |
| `apply_on_vector(vector)`      | Apply the matrix as a linear transformation on a vector | `vector: Vector`                     | `Vector`                           |
| `apply_on_matrix(matrix)`      | Apply the matrix as a linear transformation on another matrix | `matrix: Matrix`                 | `Matrix`                           |
| `compose(*matrices)`           | Compose the current matrix with one or more matrices | `*matrices: Matrix`                    | `Matrix`                           |
| `is_linear_combination(vector)`| Check if a vector is a linear combination of the columns of the matrix | `vector: Vector`              | `bool`                             |



### **Matrix Properties**
| **Property**                     | **Description**                                      |
|--------------------------------|------------------------------------------------------|
| `rank`                         | Calculate the rank of the matrix using NumPy         |
| `is_full_rank`                 | Check if the matrix is of full rank                  |
| `determinant`                  | Calculate the determinant of the matrix using NumPy  |
| `inverse`                      | Calculate the inverse of the matrix using NumPy      |
| `is_singular`                  | Check if the matrix is singular                      |
| `is_invertable`                | Check if the matrix is invertible                    |
| `is_linear_transformation`     | Check if the matrix represents a linear transformation |
| `is_linearly_dependent`        | Check if the columns of the matrix are linearly dependent | 
| `span`                         | Calculate the span of the matrix                     |
| `is_basis`                     | Check if the columns of the matrix form a basis      |
| `transpose`                    | Get the transpose of the matrix                      |
| `T`                            | Get the transpose of the matrix (alias)              |
| `diagonal`                     | Get the diagonal elements of the matrix              |
| `anti_diagonal`                | Get the anti-diagonal elements of the matrix         |
| `trace`                        | Calculate the trace of the matrix                    |
| `anti_trace`                   | Calculate the anti-trace of the matrix               |
| `is_symmetric`                 | Check if the matrix is symmetric                     |
| `is_square`                    | Check if the matrix is square                        |
| `is_diagonal`                  | Check if the matrix is diagonal                      |
| `is_anti_diagonal`             | Check if the matrix is anti-diagonal                 |
| `is_identity`                  | Check if the matrix is an identity matrix            |
| `is_upper_triangular`          | Check if the matrix is upper triangular              |
| `is_lower_triangular`          | Check if the matrix is lower triangular              |
| `is_skew_symmetric`            | Check if the matrix is skew-symmetric                |






