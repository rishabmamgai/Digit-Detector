from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import normalize
import math


def extract(data_file):

    path = f"D:\\ML\\Hanwriting ML\\data\\{data_file}"
    data = loadmat(path)

    X = normalize(data["data"], norm='max', axis=0) #/ 255  # Normalising X 
    y = data['label']

    return X.transpose(), y.transpose()
    

def rand_initialise(a, b):
    c = np.random.rand(a, b + 1) * (2 * 0.12) - 0.12
    return c


def split_data(X, y):
    m = len(y)

    train = math.ceil(m * 0.8)
    test = math.floor(m * 0.2)

    X_train = X[0:train, :]
    y_train = y[0:train, :]

    X_test = X[train:train+test, :]
    y_test = X[train:train+test, :]

    return X_train, y_train, X_test, y_test
