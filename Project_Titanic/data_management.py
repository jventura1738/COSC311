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


# Data splitting function (np train; n(p-1) testing)
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

    # These fields either don't provide any useful information, or
    # they have too many missing paramaters.
    # Cabin: way much data is missing.
    # Body: this only applies to non-survivors.
    # Ticket: this is unique to each family.
    # Boat: this only applies to survivors.
    # Home.Dest: not useful to us.
    if 'cabin' in candidates:
        new_titanic = new_titanic.drop(['cabin'], axis=1)

    if 'body' in candidates:
        new_titanic = new_titanic.drop(['body'], axis=1)

    if 'ticket' in candidates:
        new_titanic = new_titanic.drop(['ticket'], axis=1)

    if 'boat' in candidates:
        new_titanic = new_titanic.drop(['boat'], axis=1)

    if 'home.dest' in candidates:
        new_titanic = new_titanic.drop(['home.dest'], axis=1)

    if 'name' in candidates:
        new_titanic = new_titanic.drop(['name'], axis=1)

    # Fill empty ages:
    if 'age' in candidates:
        mean = new_titanic['age'].mean()
        std = new_titanic['age'].std()
        is_null = new_titanic['age'].isnull().sum()
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        age_slice = new_titanic['age'].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        new_titanic['age'] = age_slice
        new_titanic['age'] = new_titanic['age'].astype(float)

    return new_titanic


# Titanic data to a vector for K-Nearest Neighbors Algorithm.
def titanic_to_vector(titanic_data) -> List[knn_vector]:
    """ Convert a data set into a list of knn_vectors.

    Args:
        titanic_data (pd.df): The titanic dataset, or any subset of
                              such.

    Returns:
        [List[knn_vectors]]: formatted vectors for the kNN model.
    """
    # Preparing the new data.
    new_titanic = titanic_data.copy()

    # NOTE: Calls helper function, modifies titanic data.
    new_titanic = clean_titanic(new_titanic)

    # Naively fix all the data for KNN.
    genders = {'male': 0, 'female': 1}
    new_titanic['sex'] = new_titanic['sex'].map(genders)

    embarkments = {'S': 0, 'C': 1, 'Q': 2, '?': 1}
    new_titanic['embarked'] = new_titanic['embarked'].map(embarkments)

    # Prepare data:
    new_titanic['pclass'] = new_titanic['pclass'].astype(int)
    new_titanic['survived'] = new_titanic['survived'].astype(int)
    new_titanic['sibsp'] = new_titanic['sibsp'].astype(int)
    new_titanic['parch'] = new_titanic['parch'].astype(int)
    new_titanic['fare'] = new_titanic['fare'].astype(float)
    new_titanic['age'] = new_titanic['age'].astype(float)

    labels = new_titanic['survived'].to_numpy()
    dim = new_titanic.shape[1]

    # Make vectors:
    rows = list(new_titanic[0:-1].to_records(index=False))
    results = [knn_vector(dim, list(r)) for r in rows]

    # results = []
    # for r in rows:
    #     results.append(knn_vector(4, list(r)))

    return results
