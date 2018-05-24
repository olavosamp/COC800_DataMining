import numpy                as np
import pandas               as pd
import sklearn              as sk

import dirs
from load_dataset           import load_dataset
from preproc                import preproc
from sklearn.model_selection import train_test_split

from bayes                  import gaussian_naive_bayes
from show_class_splits      import show_class_splits
from least_squares          import least_squares, ridge_least_squares

testSize = 5000

dataDf, labels = load_dataset(dirs.dataset, randomState=None, numPos=10000, numNeg=10000)#fracPos=0.02, fracNeg=0.02)
dataDf = preproc(dataDf, verbose=False)
# labeledDf = dataDf.assign(Labels=labels)

trainDf, testDf, y_train, y_test = train_test_split(dataDf, labels, test_size=testSize)

print("\nTrain data loaded with following class distributions:")
show_class_splits(y_train)
print("\nTest data loaded with following class distributions:")
show_class_splits(y_test)

'Principal Components Analysis'
'   useful to reduce dataset dimensionality'
# compactDf = dimension_reduction(dataDf, keepComp=0)

'Bayesian Classifier'
bayesPred = gaussian_naive_bayes(trainDf, y_train, testDf, y_test)
print("\nNaive Bayes")
print("Correct predictions ", np.sum(bayesPred == y_test)/testSize)

## Classificadores Lineares Generalizados
'Least Squares'
lqPred = least_squares(trainDf, y_train, testDf, y_test)
print("\nLeast Squares")
print("Correct predictions ", np.sum(lqPred == y_test)/testSize)

#| com Regularização
'Least Squares with L2/Ridge Regularization'
rlqPred = ridge_least_squares(trainDf, y_train, testDf, y_test, regVal=10000)
print("\nRidge Least Squares")
print("Correct predictions ", np.sum(rlqPred == y_test)/testSize)

print("\nIs regression making a difference?\n", np.sum(lqPred == rlqPred)/testSize)

#   Regressão Logística | com Regularização
#   Regrassão Polinomial (criação de novas features)
#   Perceptron ?

#
# (K-)Nearest Neighbours (usar ball tree)
#
# Discriminador Linear      (LDA)
# Discriminador Quadrático  (QDA)
#
# Rede neural MLP
#
# Árvore de Classificação
# Ensembles
#   Bagging (Pasting algorithm)
#   Random Forest
#   Ada/Gradient Boost ou similares
#
# SVM Linear
# SVM Outros kernels
#
# Mistura Gaussiana (Não supervisionado)
