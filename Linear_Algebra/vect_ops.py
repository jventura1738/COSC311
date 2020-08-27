# Justin Ventura
# This module contains functions to perform vector operations.
# These are not very fast, but useful for understanding the operations.

from typing import List
import math

Vector = List[float] # Defining a vector as a list of floats

def add(u: Vector, v: Vector) -> Vector:
    """this function adds vectors u and v to return the resulting 
    vector w defined by u + v."""
    assert(len(u) == len(v)), 'Vectors must be of the same dimension to add.'
    return [u_i + v_i for u_i, v_i in zip(u,v)]

def subtract(u: Vector, v: Vector) -> Vector:
    """this function subtracts vectors u and v to return the resulting 
    vector w defined by u - v."""
    assert(len(u) == len(v)), 'Vectors must be of the same dimension to add.'
    return [u_i - v_i for u_i, v_i in zip(u,v)]

def summation(vectors: List[Vector]) -> Vector:
    """this function sums all vectors passed as parameters"""
    assert(vectors), 'No vectors to sum.'
    dim = len(vectors[0])
    assert all(len(v) == dim for v in vectors), 'Vectors are in different dimensions.'

    return [sum(v[i] for v in vectors) for i in range(dim)]

def scalar_multiply(scalar: float, v: Vector) -> Vector:
    """this function returns the resulting vector w defined by c * v where
    c is a scalar and v is a Vector."""
    return [scalar * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    """this function returns the component-wise means of a list of vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, summation(vectors))

def dot_product(u: Vector, v: Vector) -> float:
    """this function computes the u_1 * v_1 + ... u_n * v_n"""
    assert(len(u) == len(v)), 'Vectors must be of the same dimension.'
    return sum(u_i * v_i for u_i, v_i in zip(u, v))

def sum_of_squares(v: Vector) -> float:
    """this function computes v_1 * v_1 + ... v_n * v_n"""
    return(dot_product(v,v))

def magnitude(v: Vector) -> float:
    """this function computes length/magnitude of v"""
    return math.sqrt(sum_of_squares(v))

def distance(u: Vector, v: Vector) -> float:
    """this function computes the distance between vectors u and v"""
    return magnitude(subtract(u, v))