import numpy                as np
# import seaborn              as sns
import matplotlib.pyplot    as plt

# from mpl_toolkits.mplot3d   import Axes3D

def projection_plot(dataDf):
    # features = dataDf.shape[1]-1
    features = 10

    posData = dataDf.loc[dataDf.iloc[:, -1] == +1]  # Last DataFrame columns are labels:
    negData = dataDf.loc[dataDf.iloc[:, -1] == -1]  # used to filter data by class

    fig, axs = plt.subplots(nrows=features, ncols=features, squeeze=False)#, tight_layout=True)

    for row in range(features):
        for col in range(features):
            if row == col:
                # Plot histogram of feature i
                axs[row,col].hist(dataDf.iloc[:, col].values)  # axis.hist() method doesn't work with DataFrame
                pass
            else:
                # Plot projection X_i by X_j
                axs[row,col].plot(negData.iloc[:, row], negData.iloc[:, col], 'b.')
                axs[row,col].plot(posData.iloc[:, row], posData.iloc[:, col], 'r.')

            # Hide axis labels
            axs[row,col].get_xaxis().set_visible(False)
            axs[row,col].get_yaxis().set_visible(False)
            axs[row,col].set_yticklabels([])
            axs[row,col].set_xticklabels([])

            # Set up border labels
            if col == 0:
                axs[row,col].set_ylabel("X{}".format(row))
                axs[row,col].get_yaxis().set_visible(True)
            if row == (features-1):
                axs[row,col].set_xlabel("X{}".format(col))
                axs[row,col].get_xaxis().set_visible(True)

    plt.show()
    return axs, fig
