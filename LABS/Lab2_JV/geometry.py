# Justin Ventura COSC311
# Lab 2: geometry.py

"""
This module defines the class for geometry. Please
see the 'geometry.__doc__'
"""
from vector import R2vector

# For bookkeeping purposes.
from typing import List, Tuple

v_list = List[R2vector]
face_list = List[Tuple[R2vector, R2vector]]

class geometry:
    """
    geometry class to make a polygon of vectors.
    """
    def __init__(self, vertices: v_list = None, edges: face_list = None):
        print("hello")

g = geometry()