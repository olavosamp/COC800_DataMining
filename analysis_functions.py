import numpy                as np
import pandas               as pd

import dirs
import defines              as defs

from utils                  import get_best_thresh
from vis_functions          import plot_roc_curve

def log_reg(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train Logistic Regression classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        thres: Class discrimination threshold
    '''
    from sklearn.linear_model   import LogisticRegression

    logReg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', n_jobs=1, max_iter=100)

    logReg.fit(x_train, y_train)#, weights) # TODO: Add class weights

    if compute_threshold is True:
        probTest  = logReg.predict_proba(x_test)
        probTrain = logReg.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)

        plot_roc_curve(y_test, probTest, modelName="Logistic Regression")
    else:
        predTest    = logReg.predict(x_test)
        predTrain   = logReg.predict(x_train)

    return predTest, predTrain, logReg

def gaussian_naive_bayes(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train Naive Bayes classifier on x_train and predict on x_test
        x_train, x_test: DataFrames of shape (data x features)
    '''
    from sklearn.naive_bayes    import GaussianNB

    gnb = GaussianNB(priors=None)
    gnb.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = gnb.predict_proba(x_test)
        probTrain = gnb.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)

        plot_roc_curve(y_test, probTest, modelName="Naive Bayes")
    else:
        predTest    = gnb.predict(x_test)
        predTrain   = gnb.predict(x_train)

    return predTest, predTrain, gnb

def decision_tree(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train a Decision Tree classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''
    from sklearn.tree           import DecisionTreeClassifier

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    tree = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=15,
                                    min_samples_leaf=5)

    tree.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = tree.predict_proba(x_test)
        probTrain = tree.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = tree.predict(x_test)
        predTrain   = tree.predict(x_train)

    return predTest, predTrain, tree

def random_forest(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train an ensemble of Decision Trees on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''
    from sklearn.ensemble       import RandomForestClassifier

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    forest = RandomForestClassifier(n_estimators=50, class_weight='balanced', criterion='entropy',
                                    max_depth=15, min_samples_leaf=5, n_jobs=-1)

    forest.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = forest.predict_proba(x_test)
        probTrain = forest.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = forest.predict(x_test)
        predTrain   = forest.predict(x_train)

    return predTest, predTrain, forest

def ada_boost(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train an AdaBoost ensemble of Decision Trees on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''
    from sklearn.ensemble       import AdaBoostClassifier
    from sklearn.tree           import DecisionTreeClassifier

    tree = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=15, min_samples_leaf=5)
    ada = AdaBoostClassifier(base_estimator=tree, n_estimators=60, learning_rate=1.0)

    ada.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = ada.predict_proba(x_test)
        probTrain = ada.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)

        plot_roc_curve(y_test, probTest, modelName="AdaBoost")
    else:
        predTest    = ada.predict(x_test)
        predTrain   = ada.predict(x_train)

    return predTest, predTrain, ada

def linear_discriminant_analysis(x_train, y_train, x_test, y_test, n_components=2, compute_threshold=True):
    '''
        Train Linear Discriminant Analysis (LDA) classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        n_components: Number of components (< n_classes - 1) for dimensionality reduction.
    '''
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    lda = LinearDiscriminantAnalysis(priors=None, n_components=n_components)
    #X_r2 = lda.fit(x_train, y_train).transform(X)
    lda.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = lda.predict_proba(x_test)
        probTrain = lda.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = lda.predict(x_test)
        predTrain   = lda.predict(x_train)

    return predTest, predTrain, lda

def quadratic_discriminant_analysis(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train Quadratic Discriminant Analysis (LDA) classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        n_components: Number of components (< n_classes - 1) for dimensionality reduction.
    '''
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    qda = QuadraticDiscriminantAnalysis()
    #X_r2 = lda.fit(x_train, y_train).transform(X)
    qda.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = qda.predict_proba(x_test)
        probTrain = qda.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = qda.predict(x_test)
        predTrain   = qda.predict(x_train)

    return predTest, predTrain, qda

def nearest_neighbours(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train K-Nearest Neighbours classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
    '''
    from sklearn.neighbors     import KNeighborsClassifier

    # TODO: Experiment with 'weights' parameter
    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    knn = KNeighborsClassifier( n_neighbors=3, algorithm='ball_tree', weights='uniform',
                                p=2, metric='minkowski', n_jobs=-1)

    # print("\nParameters initialization:")
    # print(percep.coef_)

    knn.fit(x_train, y_train)#, weights) # TODO: Add class weights


    if compute_threshold is True:
        probTest  = knn.predict_proba(x_test)
        probTrain = knn.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = knn.predict(x_test)
        predTrain   = knn.predict(x_train)

    return predTest, predTrain, knn

def perceptron(x_train, y_train, x_test, y_test):
    '''
        Train a Perceptron classifier with linear activation function on x_train
        and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
    '''
    from sklearn.linear_model   import Perceptron

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    percep = Perceptron(shuffle=True, n_jobs=-1, class_weight=classWeights, max_iter=1000, tol=1e-3)

    # print("\nParameters initialization:")
    # print(percep.coef_)

    percep.fit(x_train, y_train)

    predTest = percep.predict(x_test)
    predTrain = percep.predict(x_train)

    return predTest, predTrain, percep

    # def ridge_log_reg(x_train, y_train, x_test, y_test, reg=1.0, class_weight=None):
    '''
        Train Logistic Regression classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        reg: Regularization parameter. Multiplies penalty by 1/reg, so smaller
        values mean stronger regularization.
    '''
    from sklearn.linear_model   import LogisticRegression

    logReg = LogisticRegression(penalty='l2', C=reg, solver='liblinear', n_jobs=1,
                                max_iter=100, class_weight=class_weight)

    logReg.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predictions = logReg.predict(x_test)

    return predictions, logReg
