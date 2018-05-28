import numpy                as np
import pandas               as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import dirs
import defines              as defs

def linear_discriminant_analysis(x_train, y_train, x_test, y_test, n_components=2):
    '''
        Train Linear Discriminant Analysis (LDA) classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        n_components: Number of components (< n_classes - 1) for dimensionality reduction.
    '''

    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    lda = LinearDiscriminantAnalysis(priors=None, n_components=n_components)
    #X_r2 = lda.fit(x_train, y_train).transform(X)
    lda.fit(x_train, y_train)

    predictions = lda.predict(x_test)

    return predictions, lda

def quadratic_discriminant_analysis(x_train, y_train, x_test, y_test):
    '''
        Train Quadratic Discriminant Analysis (LDA) classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        n_components: Number of components (< n_classes - 1) for dimensionality reduction.
    '''

    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    qda = QuadraticDiscriminantAnalysis()
    #X_r2 = lda.fit(x_train, y_train).transform(X)
    qda.fit(x_train, y_train)

    predictions = qda.predict(x_test)

    return predictions, qda
