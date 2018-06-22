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
from utils                  import (show_class_splits,
                                    report_performance,
                                    save_excel,
                                    save_latex)

from analysis_functions             import (gaussian_naive_bayes,
                                            log_reg,
                                            perceptron,
                                            nearest_neighbours,
                                            decision_tree, random_forest, ada_boost,
                                            linear_discriminant_analysis, quadratic_discriminant_analysis,
                                            svm, linear_svm, mlp)

from vis_functions                  import plot_conf_matrix, plot_roc_curve

if __name__ == "__main__":

    # np.set_printoptions(precision=4)

    # numPos   = 20000   # Max of    63 981 samples
    # numNeg   = 20000   # Max of 1 245 005 samples

    print("\n\n---- Loading and Preprocessing ----")

    dataDf, labels = load_dataset(dirs.dataset, randomState=defs.standardSample,
                                    fracPos=defs.fracPos, fracNeg=defs.fracNeg)
    dataDf = preproc(dataDf, verbose=False)

    testSize = round(dataDf.shape[0]*defs.fracTest)
    trainDf, testDf, y_train, y_test = train_test_split(dataDf, labels, test_size=testSize,
                                                        random_state=defs.standardSample)

    print("\nTrain data loaded with following class distributions:")
    show_class_splits(y_train)
    print("\nTest data loaded with following class distributions:")
    show_class_splits(y_test)

    # 'Principal Components Analysis'
    # '   useful to reduce dataset dimensionality'
    # compactDf = dimension_reduction(dataDf, keepComp=60)

    print("\n\n---- Classification ----\n")
    #
    # 'Naive Bayes'
    # modelName = "Naive Bayes"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = gaussian_naive_bayes(trainDf, y_train, testDf, y_test)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, conf_matrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)
    #
    #
    # 'Logistic Regression'
    # modelName = "Logistic Regression"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = log_reg(trainDf, y_train, testDf, y_test)
    #
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, conf_matrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)
    #
    #
    # 'Perceptron'
    # modelName = "Perceptron"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, _, _ = perceptron(trainDf, y_train, testDf, y_test)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, conf_matrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    # # metricsCV, conf_matrix = report_performance(y_train, predTrain, elapsed=elapsed, modelName=modelName+'_train', report=False)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)
    #
    #
    # 'Nearest Neighbours'
    # modelName = "Nearest Neighbors"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = nearest_neighbours(trainDf, y_train, testDf, y_test)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, conf_matrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)
    #
    #
    # 'Linear Discriminant Analysis'
    # modelName = "Linear Discriminant Analysis"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = linear_discriminant_analysis(trainDf, y_train, testDf, y_test, n_components=None)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, conf_matrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)
    #
    #
    # 'Quadratic Discriminant Analysis'
    # modelName = "Quadratic Discriminant Analysis"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = linear_discriminant_analysis(trainDf, y_train, testDf, y_test)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, conf_matrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)
    #
    #
    # 'Decision Tree'
    # modelName = "Decision Tree"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = decision_tree(trainDf, y_train, testDf, y_test)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, _ = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)
    #
    #
    # 'Random Forest'
    # modelName = "Random Forest"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = random_forest(trainDf, y_train, testDf, y_test)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, _ = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)
    #
    #
    # 'AdaBoost'
    # modelName = "AdaBoost"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = ada_boost(trainDf, y_train, testDf, y_test)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, confMatrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)
    #
    #
    # 'SVM'
    # modelName = "SVM"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = svm(trainDf, y_train, testDf, y_test)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, confMatrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)

    # 'Linear SVM'
    # modelName = "Linear SVM"
    # print("\n"+modelName)
    #
    # start = time.perf_counter()
    # predictions, metricsCV, _ = linear_svm(trainDf, y_train, testDf, y_test)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest, confMatrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)
    #
    # fig = plot_conf_matrix(y_test, predictions, modelName=modelName)
    #
    # save_excel(metricsTest, metricsCV)
    # save_latex(modelName)

    'MLP'
    modelName = "MLP"
    print("\n"+modelName)

    start = time.perf_counter()
    predictions, metricsCV, _ = mlp(trainDf, y_train, testDf, y_test)
    elapsed = time.perf_counter() - start

    metricsTest, confMatrix = report_performance(y_test, predictions, elapsed=elapsed, modelName=modelName, report=True)

    fig = plot_conf_matrix(y_test, predictions, modelName=modelName)

    save_excel(metricsTest, metricsCV)
    save_latex(modelName)
