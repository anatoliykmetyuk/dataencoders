import numpy as np
import pickle
import time

def shift(x, n): return np.concatenate((x[:,n:], x[:,-n:]), axis=1)

class DataSplitter(object):
  def __init__(self, length, valid_size=0.15, test_size=0.15):
    self.length = length

    self.ids = np.arange(length)
    np.random.shuffle(self.ids)

    self.valid_size = int(valid_size * length)
    self.test_size  = int(test_size  * length)

  def __call__(self, X):
    X = X[self.ids]

    test_size  = self.test_size
    valid_size = self.valid_size

    X_test  = X[:test_size]
    X_valid = X[test_size : test_size + valid_size]
    X_train = X[test_size + valid_size :]

    return X_train, X_valid, X_test


def split_data(X, y, test_and_valid_size=[0.15, 0.15]):
  s = DataSplitter(len(X), test_and_valid_size[0], test_and_valid_size[1])

  X_train, X_valid, X_test = s(X)
  y_train, y_valid, y_test = s(y)

  return X_train, y_train, X_valid, y_valid, X_test, y_test
