import numpy                as np
import pandas               as pd
import seaborn              as sns

import dirs
from preproc                import preproc
from projection_plot        import projection_plot
from corr_matrix_plot       import corr_matrix_plot
from load_dataset           import load_dataset
from dimension_reduction    import dimension_reduction
from plot_boxplot           import plot_boxplot
from plot_3d           import plot_3d

dataDf, labels = load_dataset(dirs.dataset, fracPos=0.02, fracNeg=0.002)
dataDf = preproc(dataDf, verbose=False)

## Principal Components Analysis
# useful to reduce dataset dimensionality
compactDf = dimension_reduction(dataDf, keepComp=20)

# ## Plot 3D data
# print("\n3D Projection plot")
# plot_3d(compactDf, labels)
#
# ## Custom pair plot
# print("\nPair Projection plot")
# projection_plot(compactDf.iloc[:,:10], labels)
# #
# ## Seaborn pair plot (bad)
# # sns.set(style="ticks")
# # # iris = sns.load_dataset("iris")
# # # print(iris.shape)
#
# # sns.pairplot(compactDf, hue=10)
# # plt.show()
#
# ## Correlation matrix plot
# print("\nCorrelation matrix plot")
# corr_matrix_plot(dataDf)

## Boxplot
print("\nBoxplot")
plot_boxplot(compactDf, labels)
