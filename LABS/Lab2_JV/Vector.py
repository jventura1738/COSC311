# Justin Ventura COSC311
# Lab 2: Vector.py
# OUTDATED!!!

"""
This module defines the Vector class. Please see
the class's docstring for more information.
"""

from math import sqrt

class R2vector:
    """
    This class creates an R2 vector, that is, a
    vector with an x and y component.
    Default (x,y) -> (1,1)
    """
    def __init__(self, x: float = 0, y: float = 1):
        self._x = x
        self._y = y

    # Magic Methods:
    def __repr__(self):
        return 'Vector({}, {})'.format(self._x, self._y)

    def __getitem__(self, position: int):
        return self._x if position is 0 else self._y

    # Operator Overloads.
    def __add__(self, other):
        return R2vector(self._x + other._x, self._y + other._y)

    def __sub__(self, other):
        return R2vector(self._x - other._x, self._y - other._y)

    def __eq__(self, other) -> bool:
        return (self.coords() == other.coords())

    # Class Attribute Getters.
    def get_x(self) -> float:
        """ Returns x value as float. """
        return self._x

    def get_y(self) -> float:
        """ Returns y value as float. """
        return self._y

    def coords(self) -> tuple:
        """ Returns tuple with x and y values. """
        return (self._x, self._y)

    # Vector Operations.
    def dot_prod(self, _v) -> float:
        """ Computes the dot product of self & other vector. """
        return sum(u_i * v_i for u_i, v_i in zip(self.coords(), _v.coords()))

    def sum_of_squares(self, _v) -> float:
        """ This function computes v_1 * v_1 + ... v_n * v_n """
        return(self.dot_prod(_v))

    def magnitude(self, _v) -> float:
        """ This function computes length/magnitude of v """
        return sqrt(self.sum_of_squares(_v))

    def distance(self, _v) -> float:
        """ This function computes the distance between vectors u and v """
        return self.magnitude(self - _v)
