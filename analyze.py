import numpy                as np
import pandas               as pd
import sklearn              as sk

import dirs
from load_dataset           import load_dataset
from preproc                import preproc

from bayes                  import gaussian_naive_bayes

# dataDf, labels, testDf, testLabels = load_dataset(dirs.dataset, numPos=1000, numNeg=10000)#fracPos=0.02, fracNeg=0.02)
dataDf, labels = load_dataset(dirs.dataset, numPos=1000, numNeg=10000)#fracPos=0.02, fracNeg=0.02)
dataDf = preproc(dataDf, verbose=True)

## Principal Components Analysis
# useful to reduce dataset dimensionality
# compactDf = dimension_reduction(dataDf, keepComp=0)

print(dataDf)

# newData = dataDf.drop(columns=dataDf.columns[-1])
# print(newData is dataDf)
# print(dataDf.loc[0, 0] is newData.loc[0, 0])

# bayesPred = gaussian_naive_bayes(dataDf, labels, testDf, testLabels)
# print(bayesPred)
