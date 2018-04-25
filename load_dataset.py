import numpy                as np
import pandas               as pd

def load_dataset(path, fracPos=1.0, fracNeg=1.0):
    # Load raw data
    data = np.load(path)

    # Save each class
    classPos = pd.DataFrame(data[data.keys()[0]])
    classNeg = pd.DataFrame(data[data.keys()[1]])

    # Sample a fraction of the data
    # doing this before anything else avoids memory issues
    features = classPos.shape[1]
    totPos = classPos.shape[0]
    totNeg = classNeg.shape[0]

    classPos = classPos.sample(frac=fracPos, axis=0, random_state=17)
    classNeg = classNeg.sample(frac=fracNeg, axis=0, random_state=17)

    entriesPos = classPos.shape[0]
    entriesNeg = classNeg.shape[0]
    total = entriesPos + entriesNeg

    # Create Label vectors
    yPos  =  np.ones((entriesPos,1))  # [+1, -1] representation has zero mean, which
    yNeg  = -np.ones((entriesNeg,1))  # can lead to faster gradient convergence

    # Concatenate input data and labels
    labels = np.concatenate((yPos, yNeg))

    print("\nData loaded with following class distribution: ")
    print("Positive class: {:.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))
    print("Total:          {} entries".format(total))

    # Save dataset in a DataFrame
    dataDf = pd.DataFrame(np.concatenate((classPos, classNeg)))
    dataDf.rename({-1:"labels"}, axis='columns')
    return dataDf, labels

if __name__ == "__main__":
    load_dataset()
