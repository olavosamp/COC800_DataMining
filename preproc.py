import numpy as np
import pandas as pd

def preproc(dataDf, verbose=False):
    ## Display basic dataset information, clean up and preprocess the data

    if verbose:
        print("\nData shape: ", dataDf.shape)

    # Number of zero values per feature
    zeroValues = (dataDf == 0).sum(axis=0)
    allZeros   = (dataDf == 0).all(axis=0).sum()

    # Features containing only zeros will be dropped
    dataDf = dataDf.loc[:, (dataDf != 0).any(axis=0)]
    dataDf = dataDf.reset_index(drop=True)
    print("\n{} features containing only zeros have been dropped from data.".format(allZeros))
    if verbose:
        print("\nNew data shape: ", dataDf.shape)

    # Z-score normalization
    mean = dataDf.mean(axis=0)
    std  = dataDf.std (axis=0)
    dataDf = (dataDf - mean)/std

    ## "Basic Sample Statistics"
    if verbose:
        print("\nMax:\n", dataDf.max   (axis=0).max())
        print("\nMin:\n", dataDf.min   (axis=0).min())
        print("\nMean:\n",dataDf.mean  (axis=0).mean())
        print("\nMed:\n", dataDf.median(axis=0).mean())
        print("\nVar:\n", dataDf.std   (axis=0).mean())
        print("\nStd:\n", dataDf.var   (axis=0).mean())

    return dataDf

if __name__ == "__main__":
    preproc()
