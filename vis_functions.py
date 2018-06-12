import numpy                as np
import pandas               as pd
import seaborn              as sns

import matplotlib.pyplot    as plt
import matplotlib           as mlp
from mpl_toolkits.mplot3d   import Axes3D

import defines              as defs
import dirs

mlp.rcParams['font.size'] = 24

def plot_conf_matrix(labels, predictions, modelName="", show=False, save=True, normalize=False):
    import os
    from sklearn.metrics    import confusion_matrix

    if modelName == "":
        modelName = "unnamed"

    try:
        os.makedirs(dirs.report)
    except OSError:
        pass

    fig = plt.figure(figsize=(18,10))
    confDf = pd.DataFrame(confusion_matrix(labels, predictions), index=['True E', 'True H'],
                                            columns=['Pred. E', 'Pred. H'])

    # Auto adjust colorbar limits
    vmin = None
    vmax = None
    fmt = ''

    if normalize is True:
        modelName += " normalized"

        # Normalize each row of conf matrix
        confDf.iloc[0, :] = confDf.iloc[0, :]/confDf.iloc[0, :].sum()
        confDf.iloc[1, :] = confDf.iloc[1, :]/confDf.iloc[1, :].sum()

        # Set colorbar limits to range [0, 1]
        vmin = 0.0
        vmax = 1.0
        fmt = '.2f'


    ax = sns.heatmap(confDf, annot=True, cmap='Blues', vmin=vmin, vmax=vmax, square=True, fmt=fmt,
                    annot_kws={'verticalalignment': 'center', 'horizontalalignment': 'center'})

    plt.subplots_adjust(left=0.02, bottom=0.08, right=0.90, top=0.95,
                        wspace=None, hspace=None)

    plt.title("{}".format(modelName))

    if show is True:
        plt.show()

    if save is True:
        fig.savefig(dirs.report+"Conf_matrix_"+modelName.replace(" ", "_")+".png", orientation='portrait', bbox_inches='tight')

    return ax

def plot_hyp(resultsDf, modelName, save=True, show=False):
    import os
    import ast

    try:
        os.makedirs(dirs.report)
    except OSError:
        pass

    # Convert the dict expressed in string form to dict object
    params = list(ast.literal_eval(resultsDf.loc[0, 'params']).keys())

    if len(params) == 1:
        paramName = params[0]

        x = resultsDf["param_"+paramName]

        train = resultsDf["mean_train_score"]
        test  = resultsDf["mean_test_score"]

        trainStd = resultsDf["std_train_score"]
        testStd  = resultsDf["std_test_score"]

        fig = plt.figure(figsize=(28, 15))

        # Plot train score
        plt.errorbar(x, train, yerr=trainStd,  fmt='.-', color='xkcd:tomato', markersize=8,
                    markerfacecolor='xkcd:fire engine red', markeredgecolor='xkcd:tomato',
                    ecolor="xkcd:grey green", label="Train Score")

        # Plot test score
        plt.errorbar(x, test, yerr=testStd,  fmt='.-', color='xkcd:dark blue', markersize=8,
                    markerfacecolor='xkcd:night blue', markeredgecolor='xkcd:dark blue',
                    ecolor="xkcd:grey green", label="Val Score")
        plt.legend(fontsize='small', loc='best')

        plt.ylim(ymin=0.0, ymax=1.0)             # Should be adjusted for each search space
        plt.xlim(xmin=x.min()-1, xmax=x.max()+1) # as needed

        plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
                            wspace=None, hspace=None)

        plt.title("{} parameter search: {}".format(modelName, paramName))
        plt.xlabel(paramName)
        plt.ylabel("F1 Score", fontsize=28)

        if show is True:
            plt.show()

        if save is True:
            # Save plots
            fig.savefig(dirs.report+"hyp_"+modelName.replace(" ", "_")+".pdf", orientation='portrait', bbox_inches='tight')
            fig.savefig(dirs.report+"hyp_"+modelName.replace(" ", "_")+".png", orientation='portrait', bbox_inches='tight')
    return defs.success

def projection_plot(inputDf, labels):
    '''
        inputDf is an observations by features DataFrame
        labels is an observations by 1 DataFrame of [+1, -1] labels
    '''
    features = inputDf.shape[1]
    # features = 10

    posData = inputDf[labels == +1]  # Last DataFrame columns are labels:
    negData = inputDf[labels == -1]  # used to filter data by class

    fig, axs = plt.subplots(nrows=features, ncols=features, figsize=(12,10))#, squeeze=False, tight_layout=True)

    # print("\nMax: ", features)
    for row in range(features):
        for col in range(features):
            if row <= col:
                if row == col:
                    # Plot histogram of feature i
                    axs[row,col].hist(inputDf.iloc[:, col].values, bins='auto', color='xkcd:dull blue')  # axis.hist() method doesn't work with DataFrame
                else:
                    # Plot projection X_i by X_j
                    axs[row,col].plot(posData.iloc[:, row], posData.iloc[:, col], '.', alpha=0.3, markerfacecolor='xkcd:fire engine red', markeredgecolor='xkcd:tomato')
                    axs[row,col].plot(negData.iloc[:, row], negData.iloc[:, col], '.', alpha=0.3, markerfacecolor='xkcd:night blue', markeredgecolor='xkcd:dark blue')

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


def eigen_plot(inputDf, labels):
    # Get Principal Components ratio
    eignVals =  np.linalg.svd(inputDf, compute_uv=False)
    eignVals = eignVals/eignVals.sum()

    features = len(eignVals)
    index    = list(range(features))

    # Compute cumulative contributions
    sumVals = np.zeros(features)
    for i in range(features):
        sumVals[i] = np.sum(eignVals[:i])

    sumVals = sumVals[1:]
    sumVals = np.append(sumVals, 1.0)

    # Plot barplot of Principal Components ratios
    ax = sns.barplot(x=index, y=eignVals, palette="Blues_d")

    # Plot lineplot of cumulative contributions
    ax.plot(index, sumVals, 'b-')

    ax.set_title("{} Principal values".format(features))
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.get_xaxis().set_visible(False)

    plt.show()
    return ax


def plot_3d(compactDf, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    posData = compactDf[labels == +1]
    negData = compactDf[labels == -1]

    ax.scatter(posData.iloc[:, 0], posData.iloc[:, 1], posData.iloc[:, 2], c='xkcd:fire engine red', alpha=0.3)
    ax.scatter(negData.iloc[:, 0], negData.iloc[:, 1], negData.iloc[:, 2], c='xkcd:twilight blue', alpha=0.3)

    ax.set_title("3D Scatter Plot")
    plt.show()
    return fig, ax


def corr_matrix_plot(inputDf):
    '''
    inputDf is an observations by features DataFrame
    '''
    corr = inputDf.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(13, 11))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap='viridis', vmax=.3, center=0,
                square=True, linewidths=0.00, cbar_kws={"shrink": .5})

    ax.set_title("Correlation plot of {} features".format(len(corr)))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    return fig, ax

from operator import itemgetter

def format_as_table(data,
                    keys,
                    header=None,
                    sort_by_key=None,
                    sort_order_reverse=False):
    """
    Takes a list of dictionaries, formats the data, and returns
    the formatted data as a text table.

    Required Parameters:
        data - Data to process (list of dictionaries). (Type: List)
        keys - List of keys in the dictionary. (Type: List)

    Optional Parameters:
        header - The table header. (Type: List)
        sort_by_key - The key to sort by. (Type: String)
        sort_order_reverse - Default sort order is ascending, if
            True sort order will change to descending. (Type: Boolean)
    """
    # Sort the data if a sort key is specified (default sort order
    # is ascending)
    if sort_by_key:
        data = sorted(data,
                      key=itemgetter(sort_by_key),
                      reverse=sort_order_reverse)

    # If header is not empty, add header to data
    if header:
        # Get the length of each header and create a divider based
        # on that length
        header_divider = []
        for name in header:
            header_divider.append('-' * len(name))

        # Create a list of dictionary from the keys and the header and
        # insert it at the beginning of the list. Do the same for the
        # divider and insert below the header.
        header_divider = dict(zip(keys, header_divider))
        data.insert(0, header_divider)
        header = dict(zip(keys, header))
        data.insert(0, header)

    column_widths = []
    for key in keys:
        column_widths.append(max(len(str(column[key])) for column in data))

    # Create a tuple pair of key and the associated column width for it
    key_width_pair = zip(keys, column_widths)

    format = ('%-*s ' * len(keys)).strip() + '\n'
    formatted_data = ''
    for element in data:
        data_to_format = []
        # Create a tuple that will be used for the formatting in
        # width, value format
        for pair in key_width_pair:
            data_to_format.append(pair[1])
            data_to_format.append(element[pair[0]])
        formatted_data += format % tuple(data_to_format)
    return formatted_data
