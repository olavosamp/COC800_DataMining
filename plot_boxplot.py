import numpy                as np
import pandas               as pd
import seaborn              as sns
import matplotlib.pyplot    as plt

def plot_boxplot(inputDf, labels):
    # posClass = dataDf.loc[dataDf.iloc[:, -1] == +1]
    # negClass = dataDf.loc[dataDf.iloc[:, -1] == -1]

    # ax = sns.boxplot(x='Variables', y='Values', data=dataDf.loc[:, -1].values)
    dataDf = pd.concat([inputDf, pd.DataFrame(labels, columns=['labels'])])

    ax = sns.boxplot(data=inputDf)
    ax.set_title("Boxplot of {} features".format(inputDf.shape[1]))
    plt.show()

    # Unusable, scales very badly with large number of observations
    # ax = sns.swarmplot(data=inputDf)
    # ax.set_title("Swarmplot of {} features".format(inputDf.shape[1]))
    plt.show()
    return ax
