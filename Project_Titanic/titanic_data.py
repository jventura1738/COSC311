# Justin Ventura & Blaine Mason
# Titanic Data Train-Test Module

"""
This module contains a bunch of useful functions to be used
in the process of splitting the train/test data, prepare &
clean data, and more.
"""

import numpy as np
from typing import List
from KNN_Model import knn_vector


# Data splitting function.
def partition(n: int, p: float, vectors: List[knn_vector]):
    """ Partition the given knn_vectors into training/testing groups.
        NOTE: this is done at random.
    Args:
        n [int]: number of knn_vectors.

        p [float]: percentage for the training/testing split.

        vectors [ List[knn_vector] ]: a list of all the vectors which will
                   be partitioned into two groups.

    Returns:
        test, train [ tuple( List[knn_vectors] ) ]: the partitioned data
                   in two groups with lengths np & n(1-p) respectively.
    """
    # Assertions:
    assert(vectors is not None), '[partition]: no vectors given!'
    assert(n == len(vectors)), '[partition] n is not equal to # vectors.'
    assert(0.1 <= p < 1), '[partition]: groups must be validly partitioned!'

    # Perform the partioning:
    train_size = int(n*p)
    np.random.shuffle(vectors)
    train, test = vectors[:train_size], vectors[train_size:]

    # Return the training and testing:
    return train, test
