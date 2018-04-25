import numpy                as np
import pandas               as pd
import seaborn              as sns

import dirs
from load_dataset           import load_dataset
from preproc                import preproc
from eigen_plot             import eigen_plot
from dimension_reduction    import dimension_reduction
from plot_3d                import plot_3d
from projection_plot        import projection_plot
from corr_matrix_plot       import corr_matrix_plot
from plot_boxplot           import plot_boxplot

dataDf, labels = load_dataset(dirs.dataset, fracPos=0.02, fracNeg=0.02)
dataDf = preproc(dataDf, verbose=True)

## Principal Components Analysis
# useful to reduce dataset dimensionality
compactDf = dimension_reduction(dataDf, keepComp=0)
#
# # Plot Principal Components contributions
# eigen_plot(dataDf, labels)
# #
# # ## Plot 3D data
# print("\n3D Scatter plot")
# plot_3d(compactDf.iloc[:, :3], labels)
# #
# ## Custom pair plot
print("\nPair Projection plot")
projection_plot(compactDf.iloc[:,:10], labels)
#
# ## Correlation matrix plot
# print("\nCorrelation matrix plot")
# corr_matrix_plot(dataDf)
#
# ## Boxplot
# print("\nBoxplot")
# plot_boxplot(compactDf.iloc[:, :20], labels)
