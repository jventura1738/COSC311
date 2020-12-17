# Justin Ventura COSC311
# Data Visualization Functions

"""
Color constants for visualization in the console!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sn
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


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


def load_titanic_dataset2():
    """ Import, clean, then return the Titanic Dataset """

    # Import:
    titanic = pd.read_csv('titanic_data.csv')  # Reads data nicely.
    titanic = titanic[['survived', 'fare', 'age', 'sex']]

    # Clean:
    titanic.replace({'fare': {'?': np.nan}}, regex=False, inplace=True)
    titanic.replace({'age': {'?': np.nan}}, regex=False, inplace=True)
    titanic[['age']] = titanic[['age']].astype(float)
    titanic[['fare']] = titanic[['fare']].astype(float)

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
    # Setup plot for two charts:
    colors = [(182/255, 21/255, 102/255, 0.4), (63/255, 150/255, 191/255, 0.3)]
    colors2 = [(63/255, 150/255, 1), (182/255, 21/255, 102/255, 1)]
    xtick = [0, 1]
    genders = {'male': 0, 'female': 1}
    titanic_dataset['sex'] = titanic_dataset['sex'].map(genders)
    genders = titanic_dataset['sex'].to_numpy()
    survived = titanic_dataset['survived'].to_numpy()
    data = {0:0, 1:0}
    for i in range(len(genders)):
        if survived[i] == 1:
            data[genders[i]] += 1
    # print(data)

    plt.rcParams['figure.figsize'] = [14, 6]
    plt.title('Sex Distribution')
    plt.ylabel('Number of Individuals.')
    plt.xlabel('Sex.')
    plt.xticks(xtick, ['Male', 'Female'])

    classes = dict(Counter(titanic_dataset['sex']))
    plt.bar(classes.keys(), classes.values(), color=colors)
    plt.bar(data.keys(), data.values(), color=colors2)
    plt.show()


def get_parallel(titanic_dataset):
    sex_survival = titanic_dataset[['survived', 'age', 'pclass', 'sex']]
    fig = px.parallel_coordinates(sex_survival, color="survived",
                                  labels={"Sex":'sex', "Age":'age', "Class":'pclass',
                                          "Survival":'survived'},
                                  color_continuous_scale=px.colors.sequential.Plotly3)
    fig.show()


def get_correlation_heatmap(titanic_dataset):
    # Creating a subframe that shows correlations:
    sex_survival = titanic_dataset[['survived', 'sex', 'age', 'pclass']]
    genders = {'male': 0, 'female': 1}
    sex_survival['sex'] = sex_survival['sex'].map(genders)

    # Use seaborn for the heatmap then plot it:
    sn.heatmap(sex_survival.corr(), annot=True)
    plt.show()


def sun_plot(titanic_dataset):
    sex_survival = titanic_dataset[['survived', 'age', 'pclass', 'sex']]
    fig = px.sunburst(sex_survival, path=['sex', 'pclass', 'age'],
                      values='survived')
    fig.show()


def get_parallel2(titanic_dataset):
    sex_survival = titanic_dataset[['survived', 'age', 'fare', 'sex']]
    genders = {'male': 0, 'female': 1}
    sex_survival['sex'] = sex_survival['sex'].map(genders)
    fig = px.parallel_coordinates(sex_survival, color="survived",
                                  labels={"Sex": 'sex',
                                          "Age": 'age',
                                          "Class": 'fare',
                                          "Survival": 'survived'},
                                  color_continuous_scale=px.colors.sequential.Plotly3)
    fig.show()
