# Justin Ventura COSC311
# Lab 2: Geometry.py

"""
This module defines the class for geometry. Please
see the 'Geometry.__doc__'
"""
# For book keeping purposes.
from typing import List, Dict

from vector import R2vector

# Pascal naming because my pylinter would not shut up.
VectList = List[R2vector]
EdgeDict = Dict[R2vector, List[R2vector]]

class Geometry:
    """
    geometry class to make a polygon of vectors.
    """
    def __init__(self, vertices: VectList = None, edges: EdgeDict = None):
        self._vertices = vertices
        self._edges = edges

    def add_face(self, v1: R2vector = None, v2: R2vector = None) -> None:
        """
        This function takes two vectors, then creates an edge between
        the two.  No duplicate edges allowed.
        """
        # Req 1: two R2 vectors.
        # Req 2: both R2 vectors are in the geometry.
        # Req 3: neither vectors connect already.
        assert(v1 is not None and v2 is not None), 'Expected two real vectors.'
        assert(v1 in self._vertices and v2 in self._vertices), 'One or more vector not in geometry.'
        assert(v2 not in self._edges[v1]), 'Duplicate edges not allowed!'

        self._edges[v1] += [v2]

        # [ (1,1) : [ (2,2) ] 
        #   (2,2) : [ (1,1), (3,3) ]
        #   (3,3) : [ (2,2) ]
        # ]
        # TO ADD: (3,3) to (1,1)

    # def is_closed(self, start: R2vector = None) -> bool:
    #     """
    #     Determines if the given geometry is closed or not.  That is,
    #     if there is a path that goes from start, and returns back
    #     at some point.
    #     """

    #     pass

List1 = [1]
List1 += [3]
print(List1)
