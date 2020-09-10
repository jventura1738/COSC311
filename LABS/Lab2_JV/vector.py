# Justin Ventura COSC311
# Lab 2: vector.py

"""
This module defines the vector class. Please see
the class's docstring for more information.
"""

from math import sqrt

class R2vector:
    """
    This class creates an R2 vector, that is, a
    vector with an x and y component.
    Default (x,y) -> (1,1)
    """
    def __init__(self, x: float = 1, y: float = 1):
        self._x = x
        self._y = y
        self._coords = (x, y)

    def __add__(self, other):
        return R2vector(self._x + other._x, self._y + other._y)

    def __sub__(self, other):
        return R2vector(self._x - other._x, self._y - other._y)

    def dot_prod(self, v: R2vector) -> float:
        """ Computes the dot product of self & other vector. """
        return sum(u_i * v_i for u_i, v_i in zip(self._coords, v._coords))
    
    def sum_of_squares(self, v) -> float:
        """ This function computes v_1 * v_1 + ... v_n * v_n """
        return(self.dot_prod(v))

    def magnitude(self, v: R2vector) -> float:
        """ This function computes length/magnitude of v """
        return math.sqrt(sum_of_squares(v))

    def distance(u: Vector, v: Vector) -> float:
        """ This function computes the distance between vectors u and v """
        return magnitude(subtract(u, v))
