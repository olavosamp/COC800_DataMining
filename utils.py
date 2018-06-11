import numpy        as np
import pandas       as pd

import defines      as defs
import dirs



def show_class_splits(labels):
    entriesPos = np.sum(labels == defs.posCode)
    entriesNeg = np.sum(labels == defs.negCode)
    total = len(labels)

    print("Positive class: {:5.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:5.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))
    print("Total:          {} entries".format(total))

    return defs.success


# def print_metrics(metricsDict):
#     print(metricsDict)
#     input()
#     for key in metricsDict.columns:
#         element = metricsDict[key]
#         if isinstance(element, float):
#             print("{:15}: {:.2f}".format(key, element))
#         else:
#             print("{:15}: {}".format(key, element))
#
#     return defs.success

def print_metrics(metricsDict):
    for key in metricsDict.keys():
        if isinstance(metricsDict[key], float):
            print("{:15}: {:.2f}".format(key, metricsDict[key]))
        else:
            print("{:15}: {}".format(key, metricsDict[key]))

    return defs.success


def report_performance(labels, predictions, elapsed=0, modelName="", report=True, save=True):
    from sklearn.metrics         import (f1_score, accuracy_score, roc_auc_score,
                                         precision_score, recall_score, confusion_matrix)
    from collections    import OrderedDict

    # metrics = OrderedDict()
    # metricsDf = pd.DataFrame([])
    metrics = dict()

    if modelName == "":
        modelName = "unnamed"

    metrics['Model']		= [modelName]
    metrics['Elapsed']      = [elapsed]
    metrics['accuracy']		= [accuracy_score(labels, predictions, normalize=True)]
    metrics['f1']     		= [f1_score(labels, predictions)]
    metrics['auc']			= [roc_auc_score(labels, predictions)]
    metrics['precision']	= [precision_score(labels, predictions)]
    metrics['recall']		= [recall_score(labels, predictions)]

    # metricsDf.from_dict(metrics)

    conf_matrix = confusion_matrix(labels, predictions, labels=[defs.posCode, defs.negCode])

    if report is True:
        print_metrics(metrics)
        print("\nConfusion Matrix:")
        print(conf_matrix)

    if save is True:
        modelName = modelName.replace(" ", "_")
        # Save metrics to Excel file
        metricsDf = pd.DataFrame.from_dict(metrics, orient='columns')

        filePath = dirs.report+"metrics_"+modelName+".xlsx"
        metricsDf.to_excel(filePath, index=False, float_format="%.2f")

        # Save confusion matrix to Excel file
        confDf = pd.DataFrame(conf_matrix, index=['True1', 'True2'], columns=['Predicted1', 'Predicted2'])

        filePath = dirs.report+"conf_matrix_"+modelName+".xlsx"
        confDf.to_excel(filePath, index=True)


    return metrics, conf_matrix


def save_results(cvResults, predictions, modelName):
    import os
    import dirs

    try:
        os.makedirs(dirs.results)
    except OSError:
        pass

    if not(cvResults is None):
        cvResultsPath = dirs.results+modelName.replace(" ", "_")+"_cv_results.csv"
        resultsDf = pd.DataFrame(cvResults)
        resultsDf.to_csv(cvResultsPath)
    else:
        resultsDf = None

    if not(cvResults is None):
        predPath      = dirs.results+modelName.replace(" ", "_")+"_best_pred.npy"
        np.save(predPath,  predictions)

    return resultsDf, predictions


def load_results(modelName):
    import dirs

    cvResultsPath = dirs.results+modelName.replace(" ", "_")+"_cv_results.csv"
    predPath      = dirs.results+modelName.replace(" ", "_")+"_best_pred.npy"

    resultsDf   = pd.DataFrame(pd.read_csv(cvResultsPath))
    predictions = np.load(predPath)

    return resultsDf, predictions

# def get_class_weights(y):
#     '''
#         Returns dict of class count
#     '''
#     classWeights = dict()
#     classWeights[defs.classPos] = np.sum(y == defs.classPos)
#     classWeights[defs.classNeg] = np.sum(y == defs.classNeg)
#     return classWeights
