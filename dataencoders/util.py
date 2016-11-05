import numpy as np
import pickle
import time

def shift(x, n): return np.concatenate((x[:,n:], x[:,-n:]), axis=1)

def split_data(X, y, test_and_valid_size=[0.15, 0.15]):
  size = len(X)
  ids = np.arange(len(X))
  np.random.shuffle(ids)
  X = X[ids]
  y = y[ids]

  valid_size = int(test_and_valid_size[0] * size)
  test_size  = int(test_and_valid_size[1] * size)

  def sample_data(frm, to): return X[frm:to], y[frm:to]
  X_test , y_test  = sample_data(0, test_size)
  X_valid, y_valid = sample_data(test_size, test_size + valid_size)
  X_train, y_train = sample_data(test_size + valid_size, size)

  return X_train, y_train, X_valid, y_valid, X_test, y_test
