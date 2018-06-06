import numpy                as np
import pandas               as pd
# import sklearn              as sk
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics         import f1_score, accuracy_score, roc_auc_score

import dirs
import defines              as defs
from load_dataset           import load_dataset
from preproc                import preproc, dimension_reduction
from utils                  import show_class_splits

from analysis_functions             import (gaussian_naive_bayes,
                                            log_reg, ridge_log_reg,
                                            perceptron,
                                            nearest_neighbours,
                                            decision_tree, random_forest, ada_boost,
                                            linear_discriminant_analysis, quadratic_discriminant_analysis)

# np.set_printoptions(precision=4)

# numPos   = 20000   # Max of    63 981 samples
# numNeg   = 20000   # Max of 1 245 005 samples

print("\n\n---- Loading and Preprocessing ----")

dataDf, labels = load_dataset(dirs.dataset, randomState=None, fracPos=defs.fracPos, fracNeg=defs.fracNeg)#numPos=numPos, numNeg=numNeg)
dataDf = preproc(dataDf, verbose=False)

testSize = round(dataDf.shape[0]*0.2)
trainDf, testDf, y_train, y_test = train_test_split(dataDf, labels, test_size=testSize)

print("\nTrain data loaded with following class distributions:")
show_class_splits(y_train)
print("\nTest data loaded with following class distributions:")
show_class_splits(y_test)

# 'Principal Components Analysis'
# '   useful to reduce dataset dimensionality'
# compactDf = dimension_reduction(dataDf, keepComp=60)

print("\n\n---- Classification ----\n")

'Bayesian Classifier'
print("\nNaive Bayes")
start = time.perf_counter()
bayesPred, _ = gaussian_naive_bayes(trainDf, y_train, testDf, y_test)
elapsed = time.perf_counter() - start
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(bayesPred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, bayesPred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, bayesPred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, bayesPred)))

'Logistic Regression'
print("\nLogistic Regression")
start = time.perf_counter()
logPred, _ = log_reg(trainDf, y_train, testDf, y_test)
elapsed = time.perf_counter() - start
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(logPred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, logPred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, logPred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, logPred)))

'Logistic Regression with L2 Regularization'
# TODO: Testar LogisticRegressionCV, que encontra o C otimo
logPenalty = 1/100

print("\nLogistic Regression with L2 Regularization")
start = time.perf_counter()
rlogPred, _ = ridge_log_reg(trainDf, y_train, testDf, y_test, reg=logPenalty)
elapsed = time.perf_counter() - start
print("Regularization paramenter (smaller is stronger): \n", logPenalty)
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(rlogPred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, rlogPred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, rlogPred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, rlogPred)))

'Linear Perceptron'
print("\nLinear Perceptron")
start = time.perf_counter()
percepPred, _ = perceptron(trainDf, y_train, testDf, y_test)
elapsed = time.perf_counter() - start
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(logPred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, percepPred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, percepPred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, percepPred)))

'Nearest Neighbours'
start = time.perf_counter()
knnPred, _ = nearest_neighbours(trainDf, y_train, testDf, y_test)
elapsed = time.perf_counter() - start
print("\nNearest Neighbours")
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(knnPred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, knnPred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, knnPred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, knnPred)))

'Decision Tree'
print("\nDecision Tree")
start = time.perf_counter()
treePred, _ = decision_tree(trainDf, y_train, testDf, y_test)
elapsed = time.perf_counter() - start
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(treePred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, treePred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, treePred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, treePred)))

'Random Forest'
print("\nRandom Forest")
start = time.perf_counter()
forestPred, _ = random_forest(trainDf, y_train, testDf, y_test)
elapsed = time.perf_counter() - start
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(forestPred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, forestPred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, forestPred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, forestPred)))

'AdaBoost'
print("\nAdaBoost")
start = time.perf_counter()
adaPred, _ = ada_boost(trainDf, y_train, testDf, y_test)
elapsed = time.perf_counter() - start
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(adaPred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, adaPred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, adaPred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, adaPred)))

'Linear Discriminant Analysis'
print("\nLinear Discriminant Analysis")
start = time.perf_counter()
ldaPred, _ = linear_discriminant_analysis(trainDf, y_train, testDf, y_test, n_components=None)
elapsed = time.perf_counter() - start
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(adaPred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, ldaPred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, ldaPred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, ldaPred)))

'Quadratic Discriminant Analysis'
print("\nQuadratic Discriminant Analysis")
start = time.perf_counter()
qdaPred, _ = linear_discriminant_analysis(trainDf, y_train, testDf, y_test)
elapsed = time.perf_counter() - start
print("Elapsed: {:.2f}s".format(elapsed))
# print("Correct predictions {:.4f}".format(np.sum(adaPred == y_test)/testSize))
print("Accuracy: {:.2f}".format(accuracy_score(y_test, qdaPred, normalize=True)))
print("F1 Score: {:.2f}".format(f1_score(y_test, qdaPred)))
print("AUC     : {:.2f}".format(roc_auc_score(y_test, qdaPred)))


#   Regressão Polinomial (criação de novas features)


# Rede neural MLP
#
# SVM Linear
# SVM Outros kernels
#
# Mistura Gaussiana (Não supervisionado)
