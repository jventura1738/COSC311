# Justin Ventura COSC311
# Data Visualization Functions

"""
Color constants for visualization in the console!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


# NOTE: We like dark colors.
plt.style.use('dark_background')


# -----------------------------------------------------------------------------

# TODO: make the visualizations for presentation.  These methods will perform
# calculations and print the results to the screen.  The purpose is to minimize
# the amount of code on the presentation notebook for the sake of organization.


# Load the Titanic Dataset for presentation.
def load_titanic_dataset():
    """ Import, clean, then return the Titanic Dataset """

    # Import:
    titanic = pd.read_csv('titanic_data.csv')  # Reads data nicely.
    titanic = titanic[['survived', 'pclass', 'age', 'sex']]

    # Clean:
    titanic.replace({'age': {'?': np.nan}}, regex=False, inplace=True)
    titanic[['age']] = titanic[['age']].astype(float)
    titanic[['pclass']] = titanic[['pclass']].astype(int)

    # Fix missing ages.
    mean = titanic['age'].mean()
    std = titanic['age'].std()
    is_null = titanic['age'].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    ages = titanic['age'].copy()
    ages[np.isnan(ages)] = rand_age
    titanic['age'] = ages
    titanic['age'] = titanic['age'].astype(float)

    # Return the updated dataframe.
    return titanic


# Plots the age distribution of all passengers.
def get_age_distributions(titanic_dataset):
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.title('Age Distribution')
    plt.ylabel('Number of Individuals in Age Bucket.')
    plt.xlabel('Ages Bucketed by 5.')
    plt.xticks(np.arange(0, 81, step=5))
    titanic_dataset.age.hist(bins=16, color=(1, 0.4, 0.4, 0.25))

    td = titanic_dataset.copy()
    td.drop(td[td['survived'] == 0].index, inplace=True)
    td.age.hist(bins=16, color=(1, 0.1, 0.1, 1))


# Plot the pclass 'distribution.'
def get_pclass_distributions(titanic_dataset):

    # Setup plot for two charts:
    colors = [(1, 0.4, 0.4, 0.25), (0.4, 1, 0.4, 0.25), (0.4, 0.4, 1, 0.25)]
    colors2 = [(1, 0.1, 0.1, 1), (0.1, 0.5, 0.1, 1), (0.1, 0.1, 1, 1)]

    tmp1 = titanic_dataset['pclass'].to_numpy()
    tmp2 = titanic_dataset['survived'].to_numpy()
    tmp3 = {1: 0, 2: 0, 3: 0}
    for i in range(len(tmp1)):
        if tmp2[i] == 1:
            tmp3[tmp1[i]] += 1

    plt.rcParams['figure.figsize'] = [14, 6]
    plt.ylim([0, 750])
    plt.title('Pclass Distribution')
    plt.ylabel('Number of Individuals in Pclass.')
    plt.xlabel('PClass.')
    plt.xticks(np.arange(0, 7, step=1))

    classes = dict(Counter(titanic_dataset['pclass']))
    plt.bar(classes.keys(), classes.values(), color=colors)
    plt.bar(tmp3.keys(), tmp3.values(), color=colors2)
    plt.show()


def get_sex_distributions(titanic_dataset):
    genders = {'male': 0, 'female': 1}
    titanic_dataset['sex'] = titanic_dataset['sex'].map(genders)
    genders = titanic_dataset['sex'].to_numpy()

    plt.show()


def sex_vs_survival(titanic_dataset):
    pass


def pclass_vs_survival(titanic_dataset):
    pass


def age_vs_survival(titanic_dataset):
    pass


def adultmale_vs_womenandchildren(titanic_dataset):
    pass


def get_correlation_heatmap(titanic_dataset):
    pass
