import numpy                as np
import pandas               as pd

import dirs
import defines              as defs


def log_reg(x_train, y_train, x_test, y_test):
    '''
        Train Logistic Regression classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        thres: Class discrimination threshold
    '''
    from sklearn.linear_model   import LogisticRegression

    logReg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', n_jobs=1, max_iter=100)

    logReg.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predictions = logReg.predict(x_test)

    return predictions, logReg

def ridge_log_reg(x_train, y_train, x_test, y_test, reg=1.0, class_weight=None):
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

def gaussian_naive_bayes(dataDf, labels, testDf, testLabels):
    '''
        Train Naive Bayes classifier on dataDf and predict on testDf
        dataDf, testDf: DataFrames of shape (data x features)
    '''
    from sklearn.naive_bayes    import GaussianNB

    gnb = GaussianNB(priors=None)
    gnb.fit(dataDf, labels)

    predictions = gnb.predict(testDf)

    return predictions, gnb

def decision_tree(x_train, y_train, x_test, y_test):
    '''
        Train a Decision Tree classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''
    from sklearn.tree           import DecisionTreeClassifier

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    tree = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=15, min_samples_leaf=5)

    tree.fit(x_train, y_train)

    predictions = tree.predict(x_test)

    return predictions, tree

def random_forest(x_train, y_train, x_test, y_test):
    '''
        Train an ensemble of Decision Trees on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''
    from sklearn.ensemble       import RandomForestClassifier

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    forest = RandomForestClassifier(n_estimators=50, class_weight='balanced', criterion='entropy', max_depth=15, min_samples_leaf=5, n_jobs=-1)

    forest.fit(x_train, y_train)

    predictions = forest.predict(x_test)

    return predictions, forest

def ada_boost(x_train, y_train, x_test, y_test):
    '''
        Train an AdaBoost ensemble of Decision Trees on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''
    from sklearn.ensemble       import AdaBoostClassifier
    from sklearn.tree           import DecisionTreeClassifier

    tree = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=15, min_samples_leaf=5)
    ada = AdaBoostClassifier(base_estimator=tree, n_estimators=50, learning_rate=1.0)

    ada.fit(x_train, y_train)

    predictions = ada.predict(x_test)

    return predictions, ada

def linear_discriminant_analysis(x_train, y_train, x_test, y_test, n_components=2):
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

    predictions = lda.predict(x_test)

    return predictions, lda

def quadratic_discriminant_analysis(x_train, y_train, x_test, y_test):
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

    predictions = qda.predict(x_test)

    return predictions, qda

def nearest_neighbours(x_train, y_train, x_test, y_test):
    '''
        Train K-Nearest Neighbours classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
    '''
    from sklearn.neighbors     import KNeighborsClassifier

    # TODO: Experiment with 'weights' parameter
    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    knn = KNeighborsClassifier( n_neighbors=5, algorithm='ball_tree', weights='uniform',
                                p=2, metric='minkowski', n_jobs=-1)

    # print("\nParameters initialization:")
    # print(percep.coef_)

    knn.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predictions = knn.predict(x_test)

    return predictions, knn

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

    percep.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predictions = percep.predict(x_test)

    return predictions, percep
