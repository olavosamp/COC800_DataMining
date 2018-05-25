import numpy                as np
import pandas               as pd
from sklearn.linear_model   import LogisticRegression

import dirs
import defines              as defs

def log_reg(x_train, y_train, x_test, y_test, thres=0.5):
    '''
        Train Logistic Regression classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        thres: Class discrimination threshold
    '''

    logReg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', n_jobs=-1, max_iter=100)

    logReg.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predictions = logReg.predict(x_test)

    # # Normalize predictions to [0, 1] range
    # scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
    # scaler.fit_transform(predScores.reshape(-1,1))

    # Classify test data according to threshold
    # predictions = np.where(predScores >= thres, defs.posCode, defs.negCode)

    return predictions
