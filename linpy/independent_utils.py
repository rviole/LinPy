from typing import List, Tuple, Iterable
import numbers
# Utils that are used by Vector and Matrix classes

def get_shape(data) -> Tuple[numbers.Number]:

    def can_find_length(data):
        try:
            len(data)
            return True
        except Exception as e:
            return False

    def validate_same_shapes(*data, raise_exception=True):
        shapes = [len(x) for x in data]
        if len(set(shapes)) != 1:
            if raise_exception:
                raise ValueError(
                    f"Not all shapes are same. Shapes for each element {shapes}"
                )
            return False
        return True

    def validate_same_types(*data, raise_exception=True):
        types = (type(x) for x in data)
        if len(set(types)) != 1:
            if raise_exception:
                raise TypeError(f"Not all data types in the data are same")
            return False
        return True

    def _get_shape(idx, n_rows, n_cols, ndims, *data):
        inner_data = data[0]
        if not can_find_length(inner_data):
            validate_same_types(*data, raise_exception=True)
            if ndims == 1:
                return (n_cols,)  # for 1D
            elif ndims == 0:  # for scalar
                return (0,)
            return n_rows, n_cols
        else:
            idx += 1
            ndims += 1
            if idx > 2:
                raise Exception(
                    f"The passed array has {ndims} dimensions, allowed dimensions are 1D and 2D."
                )

            validate_same_types(*data, raise_exception=True)
            validate_same_shapes(*data, raise_exception=True)
            n_cols = len(inner_data)
            n_rows = len(data)
            return _get_shape(idx, n_rows, n_cols, ndims, *inner_data)

    shape = _get_shape(0, 0, 0, 0, data)
    return shape


def zeros(shape: Tuple[int, int]) -> List[List[int]]:
    dims = len(shape)
    if dims == 1:
        return [0 for _ in range(shape[0])]
    elif dims == 2:
        return [[0 for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError("Only 1D and 2D arrays are supported")


def can_be_vector(data:Iterable) -> bool:
    try:
        
        shape = get_shape(data)
        ndim = len(shape)

        # Check if the vector is 1D
        ndim_validiton = ndim == 1

        # Check if all elements in the vector are numbers
        number_type_validation = all(isinstance(x, numbers.Number) for x in data)

        # Check if all elements in the vector are of same type
        same_type_validation = len(set([type(x).__name__ for x in data])) == 1

    except Exception as e:
        raise ValueError("Data must be an iterable")
    
    if not ndim_validiton:
        raise ValueError(f"A Vector must be 1D, got {ndim}D instead.")
    if not number_type_validation:
        raise TypeError("Vector can only contain integers or floats")
    if not same_type_validation:
        raise TypeError("Vector must contain elements of same type")

    return True


def can_be_matrix(data:Iterable[Iterable]) -> bool:
    try:
        
        shape = get_shape(data)
        ndim = len(shape)
        # Check if the matrix is 2D
        ndim_validation = ndim == 2

        # Check if all elements in the matrix are numbers
        number_type_validation = all(
            isinstance(element, numbers.Number) for row in data for element in row
        )

        # Check if all elements in the matrix are of same type
        same_type_validation_1 = len(set([type(element).__name__ for element in data])) == 1
        same_type_validation_2 = (
            len(set([type(element).__name__ for row in data for element in row])) == 1
        )
    except Exception as e:
        raise ValueError("Data must be an iterable of iterables")

    if not ndim_validation:
        raise ValueError(f"A Matrix must be 2D, got {ndim}D instead.")
    if not number_type_validation:
        raise TypeError("Matrix can only contain integers or floats")
    if not same_type_validation_1 or not same_type_validation_2:
        raise TypeError("Matrix must contain elements of same type")

    return True

