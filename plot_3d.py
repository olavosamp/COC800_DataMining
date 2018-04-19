import pandas               as pd

import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D

def plot_3d(compactDf, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(compactDf.iloc[:, 0], compactDf.iloc[:, 1], compactDf.iloc[:, 2], 'b')
    plt.show()
    return fig, ax
