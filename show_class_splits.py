import numpy                as np
import pandas               as pd

def show_class_splits(labels):
    classPos = np.sum(labels == +1)
    classNeg = np.sum(labels == -1)

    print("\nclassPos: ", classPos)
    print("classNeg: "  , classNeg)
    return
