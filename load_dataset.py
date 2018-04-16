import numpy                as np
import pandas               as pd

import dirs

def load_dataset(fracPos=1.0, fracNeg=1.0):
    # Load raw data
    data = np.load(dirs.dataset)

    # Save each class
    classPos = data[data.keys()[0]]
    classNeg = data[data.keys()[1]]

    # Create Label vectors
    yPos =  np.ones((classPos.shape[0],1))  # [+1, -1] notation has zero mean, which
    yNeg = -np.ones((classNeg.shape[0],1))  # can lead to faster gradient convergence

    # Concatenate input data and labels
    classPos = pd.DataFrame(np.concatenate((classPos, yPos), axis=1))
    classNeg = pd.DataFrame(np.concatenate((classNeg, yNeg), axis=1))

    # Sample a fraction of the data
    classPos = classPos.sample(frac=fracPos)
    classNeg = classNeg.sample(frac=fracNeg)

    entriesPos = classPos.shape[0]
    entriesNeg = classNeg.shape[0]
    total      = entriesPos + entriesNeg

    print("\nData loaded with following class distribution: ")
    print("Positive class: {:.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))

    # Save dataset in a DataFrame
    dataDf = pd.DataFrame(np.concatenate((classPos, classNeg)))
    return dataDf

if __name__ == "__main__":
    load_dataset()
