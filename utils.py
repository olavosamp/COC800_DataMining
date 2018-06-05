import numpy                as np
import pandas               as pd
import defines              as defs

def show_class_splits(labels):
    entriesPos = np.sum(labels == defs.posCode)
    entriesNeg = np.sum(labels == defs.negCode)
    total = len(labels)

    print("Positive class: {:5.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:5.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))
    print("Total:          {} entries".format(total))

    return

def print_metrics(metricsDict):
    for key in metricsDict.keys():
        if isinstance(metricsDict[key], float):
            print("{:15}: {:.2f}".format(key, metricsDict[key]))
        # elif isinstance(metricsDict[key], string)::
        else:
            print("{:15}: {}".format(key, metricsDict[key]))

    return None

# def get_class_weights(y):
#     '''
#         Returns dict of class count
#     '''
#     classWeights = dict()
#     classWeights[defs.classPos] = np.sum(y == defs.classPos)
#     classWeights[defs.classNeg] = np.sum(y == defs.classNeg)
#     return classWeights
