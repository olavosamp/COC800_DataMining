import pandas               as pd

import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D

def plot_3d(compactDf, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    posData = compactDf[labels == +1]
    negData = compactDf[labels == -1]

    ax.scatter(posData.iloc[:, 0], posData.iloc[:, 1], posData.iloc[:, 2], c='xkcd:twilight blue', alpha=0.3)
    ax.scatter(negData.iloc[:, 0], negData.iloc[:, 1], negData.iloc[:, 2], c='xkcd:fire engine red', alpha=0.3)

    ax.set_title("3D Scatter Plot")
    plt.show()
    return fig, ax
