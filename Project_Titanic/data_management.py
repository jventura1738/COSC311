# Justin Ventura & Blaine Mason
# Data Management specifically for Titanic Data.

"""
This module contains useful functions for data preparation
cleaning, splitting, and management.
"""

# import re  # Used for regex filter.
import numpy as np  # Ol' Reliable.
# import pandas as pd  # Dataframes.
from typing import List  # Type annotations.
from KNN_Model import knn_vector  # KNN Model.


# Function to clean/prune titanic data.
def clean_titanic(titanic_data):
    """ Fixes missing values and removes unnessary data.

    Args:
        titanic_data (pd.df): the un-cleaned dataset.

    Returns:
        [pd.df]: a copy of the new version of the dataset.

    NOTE: Changes 'cabin' to 'deck.'
    """
    candidates = np.array(titanic_data.columns)
    new_titanic = titanic_data.copy()

    if 'cabin' in candidates:
        new_titanic = new_titanic.drop(['cabin'], axis=1)

    if 'body' in candidates:
        new_titanic = new_titanic.drop(['body'], axis=1)

    if 'ticket' in candidates:
        new_titanic = new_titanic.drop(['ticket'], axis=1)

    if 'boat' in candidates:
        new_titanic = new_titanic.drop(['boat'], axis=1)

    if 'age' in candidates:
        mean = new_titanic['age'].mean()
        std = new_titanic['age'].std()
        is_null = new_titanic['age'].isnull().sum()
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        age_slice = new_titanic['age'].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        new_titanic['age'] = age_slice
        new_titanic['age'] = new_titanic['age'].astype(float)

    for c in candidates:
        print('gamer')

    return new_titanic


# Titanic data to a vector for K-Nearest Neighbors Algorithm.
def titanic_to_vector(titanic_data) -> List[knn_vector]:
    """ Convert a data set into a list of knn_vectors.

    Args:
        titanic_data (pd.df): The titanic dataset, or any subset of
                              such.

    Returns:
        [List[knn_vectors]]: formatted vectors for the kNN model.

        [integer]: dimension of the vector.
    """
    candidates = np.array(titanic_data.columns)
    for c in candidates:
        print(f'{c} has type :{type(c)}')
        if c == 'pclass':
            print('do the pclass')
        elif c == 'survived':
            print('do the survived')
        elif c == 'sex':
            print('do the sex')
        elif c == 'age':
            print('do the age')
        elif c == 'sisb':
            print('do the sibsp')
        elif c == 'parch':
            print('do the parch')
        elif c == 'fare':
            print('do the fare')
        elif c == 'sex':
            print('do the sex')
    pass


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
