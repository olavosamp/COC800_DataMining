import numpy                as np
import pandas               as pd

import dirs

def load_dataset():
    # Load raw data
    data = np.load(dirs.dataset)

    # Save each class
    classA = data[data.keys()[0]]
    classB = data[data.keys()[1]]

    # Create Label vectors
    elemsClassA = classA.shape[0]
    elemsClassB = classB.shape[0]

    yA =  np.ones((elemsClassA,1))
    yB = -np.ones((elemsClassB,1))

    # Concatenate input data and labels
    classB = np.concatenate((classB, yB), axis=1)
    classA = np.concatenate((classA, yA), axis=1)

    # print("Positive class shape: ", classA.shape)
    # print("Negative class shape: ", classB.shape)

    # Save dataset in a DataFrame
    dataDf = pd.DataFrame(np.concatenate((classA, classB)))
    return dataDf
