import numpy                as np
import pandas               as pd

import dirs

def load_dataset(fracPos=1.0, fracNeg=1.0):
    # Load raw data
    data = np.load(dirs.dataset)

    # Save each class
    # classPos = pd.DataFrame(data[data.keys()[0]])
    # classNeg = pd.DataFrame(data[data.keys()[1]])
    classPos = data[data.keys()[0]]
    classNeg = data[data.keys()[1]]

    # Sample a fraction of the data
    # doing this before anything else avoids memory issues
    totPos = classPos.shape[0]
    totNeg = classNeg.shape[0]

    # choose random indexes
    indexPos = np.random.choice(range(totPos), size=int(totPos*fracPos), replace=False)
    indexNeg = np.random.choice(range(totNeg), size=int(totNeg*fracNeg), replace=False)

    classPos = classPos[indexPos,:]
    classNeg = classNeg[indexNeg,:]

    # classPos = classPos.sample(frac=fracPos, axis=0)
    # classNeg = classNeg.sample(frac=fracNeg, axis=0)

    entriesPos = classPos.shape[0]
    entriesNeg = classNeg.shape[0]
    total = entriesPos + entriesNeg

    # Create Label vectors
    yPos  =  np.ones((entriesPos,1))  # [+1, -1] representation has zero mean, which
    yNeg  = -np.ones((entriesNeg,1))  # can lead to faster gradient convergence

    # Concatenate input data and labels

    classPos = pd.DataFrame(np.concatenate((classPos, yPos), axis=1))
    classNeg = pd.DataFrame(np.concatenate((classNeg, yNeg), axis=1))

    print("\nData loaded with following class distribution: ")
    print("Positive class: {:.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))

    # Save dataset in a DataFrame
    dataDf = pd.DataFrame(np.concatenate((classPos, classNeg)))
    return dataDf

if __name__ == "__main__":
    load_dataset()
