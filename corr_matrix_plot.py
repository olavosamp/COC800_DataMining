import numpy                as np
import seaborn              as sns
import matplotlib.pyplot    as plt

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
