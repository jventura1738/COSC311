# Justin Ventura

import matplotlib.pyplot as plt    # Matplotlib as usual.
import plotly.express as px
import pandas as pd


# Display the clusters.
def display_cluster(vectors):
    # Show Training Data:
    # for v in vectors:
    #     print(v.get_values())

    X = [v.get_values()[0] for v in vectors]
    Y = [v.get_values()[1] for v in vectors]
    Z = [v.get_values()[2] for v in vectors]

    df = pd.DataFrame()
    df['x'] = X
    df['y'] = Y
    df['z'] = Z
    print(df)

    fig = px.scatter_3d(df, x='x', y='y', z='z')
    fig.show()
