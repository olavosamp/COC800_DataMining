import numpy                as np
# import seaborn              as sns
import matplotlib.pyplot    as plt

from sklearn.decomposition  import PCA
from mpl_toolkits.mplot3d   import Axes3D

from preproc                import preproc
from projection_plot        import projection_plot

dataDf = preproc()

## Principal Components Analysis
# dataPCA = PCA(n_components=None)
# xCompact = dataPCA.fit_transform(dataDf.iloc[:, :-1])
#
# explVar = dataPCA.explained_variance_ratio_
# print("\nExplained variance:\n", explVar)
# print("\nN components:\n", dataPCA.n_components_)
#
# print("\nPrincipal components: ", np.shape(dataPCA.components_)[0])
# print("\nCompact data: ", np.shape(xCompact))

## Plot data
# plt.plot(xCompact[:, 0], xCompact[:, 1], 'b.')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xCompact[:, 0], xCompact[:, 1], xCompact[:, 2], 'b')
# plt.show()

projection_plot(dataDf)
