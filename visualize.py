import numpy                as np
import seaborn              as sns
import matplotlib.pyplot    as plt

from keras.datasets         import mnist

(x_train, y_train), (x_test, y_test)  = mnist.load_data()

print(np.shape(x_train))
