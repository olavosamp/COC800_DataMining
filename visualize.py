import numpy                as np
import pandas               as pd
import seaborn              as sns
import matplotlib.pyplot    as plt

from sklearn.decomposition  import PCA
from mpl_toolkits.mplot3d   import Axes3D

import dirs
from preproc                import preproc
from projection_plot        import projection_plot
from corr_matrix_plot       import corr_matrix_plot
from load_dataset           import load_dataset
from dimension_reduction    import dimension_reduction

dataDf = load_dataset(dirs.dataset, fracPos=0.2, fracNeg=0.02)
dataDf = preproc(dataDf, verbose=True)

## Principal Components Analysis
# useful to reduce dataset dimensionality
compactDf = dimension_reduction(dataDf, keepComp=50)

## Plot 3D data
print("\n3D Projection plot")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(compactDf.iloc[:, 0], compactDf.iloc[:, 1], compactDf.iloc[:, 2], 'b')
plt.show()

## Custom pair plot
print("\nPair Projection plot")
projection_plot(compactDf.iloc[:,:10], compactDf.iloc[:,-1])

## Seaborn pair plot (bad)
# sns.set(style="ticks")
# # iris = sns.load_dataset("iris")
# # print(iris.shape)

# sns.pairplot(compactDf, hue=10)
# plt.show()

## Correlation matrix plot
print("\nCorrelation matrix plot")
corr_matrix_plot(dataDf.iloc[:, :-1])
