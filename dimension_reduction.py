import numpy                as np
import pandas               as pd

from sklearn.decomposition  import PCA

def dimension_reduction(dataDf, keepComp=0):
    '''
    dataDf:   is an observations by features DataFrame
    keepComp: Number of components to keep
    '''
    if keepComp <= 0:
        keepComp = dataDf.shape[1]

    dataPCA = PCA(n_components=None)

    xCompact = dataPCA.fit_transform(dataDf)

    explVar = dataPCA.explained_variance_ratio_
    # print("\nExplained variance:\n", explVar)
    print("\nN components:", dataPCA.n_components_)
    print("\nPrincipal components to keep: ", keepComp)

    # # Save compact data as the reference DataFrame
    # yCol = dataDf.iloc[:, -1].values.reshape((dataDf.iloc[:, -1].shape[0], 1))
    # compactDf = pd.DataFrame(np.concatenate((xCompact[:, :keepComp], yCol), axis=1))
    compactDf = pd.DataFrame(xCompact[:, :keepComp])
    print("\nCompact data: ", np.shape(compactDf))
    return compactDf