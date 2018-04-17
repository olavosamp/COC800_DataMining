import numpy as np
import pandas as pd

def preproc(dataDf, verbose=False):
    ## Display basic dataset information, clean up and preprocess the data

    if verbose:
        print("\nData shape: ", dataDf.shape)

    # Number of zero values per feature
    zeroValues = (dataDf.iloc[:,:-1] == 0).sum(axis=0)
    allZeros   = (dataDf.iloc[:,:-1] == 0).all(axis=0).sum()
    # print("\nNumber of zero-valued entries per feature:\n", zeroValues)

    # Features containing only zeros will be dropped
    dataDf = dataDf.loc[:, (dataDf != 0).any(axis=0)]
    print("\n{} features containing only zeros have been dropped from data.".format(allZeros))

    if verbose:
        print("\nNew data shape: ", dataDf.shape)

    # Z-score normalization
    mean = dataDf.iloc[:, :-1].mean(axis=0)
    std  = dataDf.iloc[:, :-1].std (axis=0)
    dataDf.iloc[:, :-1] = (dataDf.iloc[:, :-1] - mean)/std

    ## "Basic Sample Statistics"
    if verbose:
        print("\nMax:\n", dataDf.iloc[:, :-1].max   (axis=0))
        print("\nMin:\n", dataDf.iloc[:, :-1].min   (axis=0))
        print("\nMean:\n",dataDf.iloc[:, :-1].mean  (axis=0))
        print("\nMed:\n", dataDf.iloc[:, :-1].median(axis=0))
        print("\nVar:\n", dataDf.iloc[:, :-1].std   (axis=0))
        print("\nStd:\n", dataDf.iloc[:, :-1].var   (axis=0))

    return dataDf

if __name__ == "__main__":
    preproc()
