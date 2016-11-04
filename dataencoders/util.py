import numpy as np
import pickle
import time

def loadLines(fname):
  with open(fname) as f:
    content = f.readlines()
    return np.array(content, dtype=object)

def load(fname):
  with open(fname) as f:
    return f.read()

def save_model(path_arr, path_ids, arr, mappings):
  with open(path_arr, 'wb') as f_arr, open(path_ids, 'wb') as f_ids:
    np.save(f_arr, arr)
    pickle.dump(mappings, f_ids)

def load_model(path_arr, path_ids):
  with open(path_arr, 'rb') as f_arr, open(path_ids, 'rb') as f_ids:
    return np.load(f_arr), pickle.load(f_ids)

def benchmark(t, str=None):
  if t['t']:
    t_taken = time.time() - t['t']
    print('Time taken', t_taken)
  t['t'] = time.time()

  if str:
    print(str)

def time_tracker(): return {'t': None}

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
