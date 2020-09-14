# Justin Ventura COSC311
# Lab 2: Geometry.py
# OUTDATED!!!

"""
This module defines the class for geometry. Please
see the 'Geometry.__doc__'
"""
# For book keeping purposes.
from typing import List, Tuple

from Vector import R2vector as Vector

# Pascal naming because my pylinter would not shut up.
VectList = List[Vector]
EdgeList = List[Tuple[Vector, Vector]]

class Geometry:
    """
    geometry class to make a polygon of vectors.
    """
    # Magic Methods:
    def __init__(self, vertices: VectList, edges: EdgeList):
        self._vertices = vertices
        self._edges = edges

    def __str__(self):
        return f'Vertices: {self._vertices}.\n Edges: {self._edges}'
    
    # Utility Methods:

    def display(self) -> None:
        print(f'Vertices: {self._vertices}.\n Edges: {self._edges}')

    def add_face(self, From: Vector = None, To: Vector = None) -> None:
        """
        This function takes two vectors, then creates an edge between
        the two.  No duplicate edges allowed.
        """

        # Req 1: vector 2 exists.
        assert(To is not None), 'Adding a face requires two vectors.'
        # Req 2: neither vectors connect already.
        assert((From, To) not in self._edges), 'Duplicate edges not allowed!'

        self._edges.append((From, To))

        # EXAMPLE:
        # [ (1,1) : [ (2,2) ]
        #   (2,2) : [ (1,1), (3,3) ]
        #   (3,3) : [ (2,2) ]
        # ]
        # TO ADD: (3,3) to (1,1)
        return None

    def is_closed(self, origin: Vector = None) -> bool:
        """
        Determines if the given geometry is closed or not.  That is,
        if there is a path that goes from start, and returns back
        at some point.
        """
        pass

# Testing values:
my_list = [[Vector(i, i), Vector(i+1, i+1)] for i in range(10)]
my_list += [[Vector(10+i, 10-i), Vector(10+i+1, 10-i-1)] for i in range(10)]
my_list.append([Vector(20, 0), (0,0)])
# for elem in my_list:
#     print(elem)

# Setting up the test polygon:
all_vertices = [x for x, _ in my_list]
poly = Geometry(all_vertices, my_list)
print(poly)
