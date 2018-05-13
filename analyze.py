import numpy                as np
import pandas               as pd
import sklearn              as sk

import dirs
from load_dataset           import load_dataset
from preproc                import preproc
from sklearn.model_selection import train_test_split

from bayes                  import gaussian_naive_bayes
from show_class_splits      import show_class_splits

testSize = 1000



dataDf, labels = load_dataset(dirs.dataset, numPos=10000, numNeg=10000)#fracPos=0.02, fracNeg=0.02)
dataDf = preproc(dataDf, verbose=True)
# labeledDf = dataDf.assign(Labels=labels)

trainDf, testDf, y_train, y_test = train_test_split(dataDf, labels, test_size=testSize)

show_class_splits(labels)
show_class_splits(y_train)
show_class_splits(y_test)

'Principal Components Analysis'
'   useful to reduce dataset dimensionality'
# compactDf = dimension_reduction(dataDf, keepComp=0)

'Bayesian Classifier'
bayesPred = gaussian_naive_bayes(trainDf, y_train, testDf, y_test)
print("\nCorrect predictions ", np.sum(bayesPred == y_test)/testSize)
