import numpy as np
import pandas as pd

from load_dataset import load_dataset

dataDf = load_dataset()

# Basic dataset information
print("\nDF Shape: ", dataDf.shape)

# Number of zero values per feature
zeroValues = (dataDf.iloc[:,:-1] == 0).sum(axis=0)
allZeros   = (dataDf.iloc[:,:-1] == 0).all(axis=0).sum()
print("\nNumber of zero-valued entries per feature:\n", zeroValues)

# Features containing only zeros will be dropped
dataDf = dataDf.loc[:, (dataDf != 0).any(axis=0)]
print("\n{allZeros} features containing only zeros have been dropped from data.")

print("\nNew data shape: ", dataDf.shape)
print(dataDf.head())
