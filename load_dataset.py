import numpy                as np

import dirs


data = np.load(dirs.dataset)

print("0 shape: ", data[data.keys()[0]].shape)
print("1 shape: ", data[data.keys()[1]].shape)
