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
from visualizations import BOLD, ENDC  # Visualization.


# Data splitting function (np train; n(p-1) testing)
def partition(n: int, p: float, vectors: List[knn_vector], rand: bool = False):
    """ Partition the given knn_vectors into training/testing groups.
        NOTE: this is done at random.
    Args:
        n [int]: number of knn_vectors.

        p [float]: percentage for the training/testing split.

        vectors [ List[knn_vector] ]: a list of all the vectors which will
                be partitioned into two groups.

        rand [bool, optional]: random partitions, defaults to False.

    Returns:
        train, test [ tuple( List[knn_vectors] ) ]: the partitioned data
                    in two groups with lengths np & n(1-p) respectively.
    """
    # Assertions:
    assert(vectors is not None), '[partition]: no vectors given!'
    assert(n == len(vectors)), '[partition] n is not equal to # vectors.'
    assert(0.1 <= p < 1), '[partition]: groups must be validly partitioned!'

    # Perform the partioning:
    train_size = int(n*p)
    if rand:
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

    # if 'fare' in candidates:
    #     new_titanic = new_titanic.drop(['fare'], axis=1)

    # Fill empty ages:
    if 'age' in candidates:
        mean = new_titanic['age'].mean()
        std = new_titanic['age'].std()
        is_null = new_titanic['age'].isnull().sum()
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        ages = new_titanic['age'].copy()
        ages[np.isnan(ages)] = rand_age
        new_titanic['age'] = ages
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

    embarkments = {'S': 0, 'C': 1, 'Q': 2, '?': 0}
    new_titanic['embarked'] = new_titanic['embarked'].map(embarkments)

    # Prepare data:
    new_titanic['pclass'] = new_titanic['pclass'].astype(int)
    new_titanic['survived'] = new_titanic['survived'].astype(int)
    new_titanic['sibsp'] = new_titanic['sibsp'].astype(int)
    new_titanic['parch'] = new_titanic['parch'].astype(int)
    new_titanic['fare'] = new_titanic['fare'].astype(float)
    new_titanic['age'] = new_titanic['age'].astype(float)

    # Lets make a new column!
    genders, ages = new_titanic['sex'].to_list(), new_titanic['age'].to_list()
    adult_male = [1] * new_titanic.shape[0]

    i = 0
    for g, a in zip(genders, ages):
        if g == 0 and a >= 18:
            adult_male[0] = 1
        i += 1

    new_titanic['adult_male'] = adult_male  # New column.

    # Reorder the dataset.
    cols = new_titanic.columns.to_list()
    target = int(cols.index('survived'))
    cols.pop(target)
    cols.append('survived')

    new_titanic = new_titanic[cols]
    dim = new_titanic.shape[1] - 1

    new_titanic['age'] = new_titanic['age'] / 10.0
    new_titanic['fare'] = new_titanic['fare'] / 10.0

    # Convert the dataset to rows into KNN vectors.
    rows = list(new_titanic.to_records(index=False))
    results = [knn_vector(dim, list(r)) for r in rows]

    # Return the results.
    return results


# KNN Prediction Function:
def titanic_KNN(kmodel, kvects, verbose=False) -> None:
    """K-Nearest Neighbors Prediction.

    Args:
        kmodel (kNN_Model): Performs KNN Prediction with this model.
        kvect (List[knn_vector]): List of vector for KNN.
        verbose (bool, optional): For logging. Defaults to False.

    Returns:
        float : precision percentage score of the trial.
    """
    n = len(kvects)
    assert(n > 0), 'Number of vectors must be positive!'

    if verbose and n > 600:
        print(f'[!] -> {n} rows may take a few moments...')

    # These keep track of accuracy.
    TS = TN = FS = FN = 0
    correct = 0

    predicted = 0

    # Iterate through and test points:
    for v in kvects:
        true_label = int(v.get_label())
        result = int(kmodel.predict(v.get_values()))

        if result == 1:
            predicted += 1

        if result == true_label:
            correct += 1
            if result == 1:
                TS += 1
            if result == 0:
                TN += 1
        else:
            if result == 1:
                FS += 1
            if result == 0:
                FN += 1

    print('\n λ Confusion Matrix λ', )
    print('-'*20)
    print(f'n={n} expected')
    print('         S   NS')
    print(f'model S  {TS} {FS} -> {TS + FS}')
    print(f'guess NS {FN} {TN} -> {FN + TN}')
    print('          v   v')
    print(f'        {TS+FN} {FS+TN}')
    print('-'*20)

    if verbose:
        print('STATS: ')
        print(f'Numerical Error: {abs(n - correct)}')
        print(f'Precision: {BOLD}{correct/n * 100}%{ENDC}\n')

    # Return the accuracy percentage.
    return correct/n * 100


# Titanic prediction function.
def titanic_predictions(model=None, test_vects=None, verbose=False) -> None:
    """ Predicts Titanic Data based on model presented.

    Args:
        model (ML Model): The KNN Model.
        test_vects (knn_vector): [description]
        verbose (bool, optional): For logging. Defaults to False.

    Returns:
        float : precision percentage score of the ML trial.
    """
    assert(model is not None), 'Machine Learning Model must be provided!'

    # K-Nearest Neighbor Predictions:
    if model.name == 'kNN_Model':
        if verbose:
            print('[!] -> Logging K-Nearest Neighbors Predictions.')

        score = titanic_KNN(kmodel=model, kvects=test_vects, verbose=verbose)

        if verbose:
            print('[!] -> End Logging K-Nearest Neighbors Predictions.')

    # Others
    else:
        print('do nothing')

    return score
