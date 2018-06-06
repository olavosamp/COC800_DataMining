import numpy        as np
import pandas       as pd
import defines      as defs

from collections    import OrderedDict

def show_class_splits(labels):
    entriesPos = np.sum(labels == defs.posCode)
    entriesNeg = np.sum(labels == defs.negCode)
    total = len(labels)

    print("Positive class: {:5.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:5.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))
    print("Total:          {} entries".format(total))

    return defs.success

def print_metrics(metricsDict):
    for key in metricsDict.keys():
        if isinstance(metricsDict[key], float):
            print("{:15}: {:.2f}".format(key, metricsDict[key]))
        else:
            print("{:15}: {}".format(key, metricsDict[key]))

    return defs.success

def report_performance(labels, predictions, elapsed=0, model_name="", report=True):
    from sklearn.metrics         import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
    metrics = OrderedDict()

    metrics['Elapsed']      = elapsed
    metrics['Model']		= model_name
    metrics['accuracy']		= accuracy_score(labels, predictions, normalize=True)
    metrics['f1']     		= f1_score(labels, predictions)
    metrics['auc']			= roc_auc_score(labels, predictions)
    metrics['precision']	= precision_score(labels, predictions)
    metrics['recall']		= recall_score(labels, predictions)

    if report == True:
        print_metrics(metrics)

    return metrics

# def get_class_weights(y):
#     '''
#         Returns dict of class count
#     '''
#     classWeights = dict()
#     classWeights[defs.classPos] = np.sum(y == defs.classPos)
#     classWeights[defs.classNeg] = np.sum(y == defs.classNeg)
#     return classWeights
