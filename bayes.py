import numpy                as np
import pandas               as pd
from sklearn.naive_bayes    import GaussianNB

import dirs

def gaussian_naive_bayes(dataDf, labels, testDf, testLabels):
    gnb = GaussianNB()
    gnb.fit(dataDf, labels)

    # print(dataDf.shape)
    # print(testDf.shape)


    predictions = gnb.predict(testDf)

    return predictions
