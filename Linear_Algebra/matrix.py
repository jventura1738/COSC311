# Justin Ventura
# This module includes functions to create matrices.

from typing import Tuple, List, Callable

Vector = List[float]
Matrix = List[List[float]]
A=[ [1,2],
    [3,4],
    [5,6] ]

B=[ [1, 2, 3],
    [4, 5, 6]]

def shape(A: Matrix) -> Tuple[int, int]:
    """returns the shape of the matrix"""
    rows = len(A)
    cols = len(A[0]) if A else 0
    return rows, cols # as a tuple

def get_row(A: Matrix, i: int) -> Vector:
    """returns the i-th row as a vector"""
    return A[i]

def get_col(A: Matrix, j: int) -> Vector:
    """returns the i-th col as a vector"""
    return [A_i[j] for A_i in A]

def make_matrix(rows: int, cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """returns a rows * cols matrix whose (i, j)-th entry is entry_fn(i, j)"""
    return[[entry_fn(i, j) for j in range(cols)] for i in range(rows)]

def identity_matrix(dim: int) -> Matrix:
    """returns the identity matrix"""
    return make_matrix(dim, dim, lambda i, j: i if i==j else 0)