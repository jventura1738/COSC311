# Justin Ventura & Blaine Mason
# K-Nearest Neighbors Model Module

"""
This module contains the K-Nearest Neighbor Model Class
that is one of the few Machine Learning Algorithms for
the Titanic Prediction Project.
"""

import numpy as np  # Ol' Reliable
import numpy.linalg as la  # Linear Alg np.
from typing import List  # For type hinting.
from scipy import stats  # Makes my life easier


# Vector Class for the k-Nearest Neighbor Class.
class knn_vector:
    """ k-Nearest Neighbor vector class.

    This converts the format:
    List[ ( vals ... class ), ... ]

    into...
    Vector( <data>, <label> )
    """
    # Constructor of the vector class.
    def __init__(self, dim: int, vals):
        """ Initializes the vector. [Constructor]

        Args:
            dim [int]: number of quantitative elements in the list.
            vals [list]: the actual list for quantitative data + the class.
        """
        assert(dim and vals), '[init k_vect]: missing paramters!'
        assert(len(vals) - 1 == dim), f'[init k_vect]: {dim} != len(values).'

        # Create the vector.
        self.dimensions = dim
        self.values = np.array(vals[0:-1])
        self.label = vals[-1]

    # Getter function for the values.
    def get_values(self):
        """ Return the values of the vector (no label).

        Args:
            None.  Just a simple getter.

        Returns:
            self.values [np.array(float/int)]: simple getter.
        """
        return self.values

    # Getter function for the label.
    def get_label(self):
        """ Return the label of the vector.

        Args:
            None.  Just a simple getter.

        Returns:
            self.label [any, typically int]: simple getter.
        """
        return self.label

    # Dunder for length:
    def __len__(self) -> int:
        """ Dunder for length.

        Returns self.dimensions as len(<knn_vector>).
        """
        return self.dimensions

    # Dunder for printing:
    def __str__(self):
        """ Dunder for printing.

        Returns formatted string -> Vect: <vector>, Label: <label>.
        """
        return f'Vect: {self.values}, Label: {self.label}'


# K-Nearest Neighbor Class.
class kNN_Model:
    """ k-Nearest Neighbors Model Class.

    Uses training data in order to predict future values
    of new query points.
    """
    # kNN Model Constructor.
    def __init__(self, train_vect: List[knn_vector], k: int = 3) -> None:
        """ Initializes the training_data. [Constructor]

        Args:
            k [int]: the number of neighbors the model should
                     use in the 'voting' system. 3 by default.

            train_x [list]: pre-labeled (class) training data.

            -> train_x format: list[list[tuple(<numeric-data>), class]]
        """
        self.k = k
        self.training_vect = train_vect

    # Calculate the distances of each train vector from the test vector.
    def distances(self, test_v=None) -> list:
        """ Calculates the distance between the training data & test point.
        NOTE: done in linear O(n) time.

        Args:
            test_v [tuple]: a specific value to be tested against the training
            data.
        """
        return [la.norm(t.get_values() - test_v, axis=0)
                for t in self.training_vect]

    # Give the model training data.
    def train(self, train_v: List[knn_vector] = None) -> None:
        """ Trains the model with pre-labelled data.

        Args:
            train_x [List[knn_vector]]: the training data used to train model.
            * NOTE: if train_v is None, function immediately returns. *

            -> train_x format: list[list[tuple(<numeric-data>), class]]

        Returns:
            None.
        """
        if not self.training_vect:
            self.training_vect = train_v

    # Predict the value of a given query point; returns label.
    def predict(self, query_point=None):
        """ Uses the model in order to predict what to label the query point.

        Args:
            query_point [tuple(<numericdata>)]: the point in which a class will
            be predicted for.

        Returns:
            'Label' [int]: The label will be the num class that this algorithm
                           deems closest to the possible label its been trained
                           to identify.
        """
        # TODO: use a k-d tree for O(logn) searches.  For now we can use the
        # naive O(n) "brute force" approach.
        assert(query_point is not None), '[predict()]: query point undefined.'

        # Get the top k vectors with lowest distances:
        labels = [v.get_label() for v in self.training_vect]
        dist = zip(self.distances(query_point), labels)
        dist = sorted(dist)

        # Get the 'votes' from the candidates and return the 'winner.'
        candidates = [d[1] for d in dist[:self.k]]
        return stats.mode(candidates)[0]


# Main for directly running the script.
if __name__ == '__main__':
    print('This is the k nearest neighbor model module! :)')
    print('-Justin Ventura λ')
