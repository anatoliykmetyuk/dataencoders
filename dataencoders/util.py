import numpy as np
import pickle
import time

def save_model(path_arr, path_ids, arr, mappings):
  with open(path_arr, 'wb') as f_arr, open(path_ids, 'wb') as f_ids:
    np.save(f_arr, arr)
    pickle.dump(mappings, f_ids)

def load_model(path_arr, path_ids):
  with open(path_arr) as f_arr, open(path_ids) as f_ids:
    return np.load(f_arr), pickle.load(f_ids)

def benchmark(t, str=None):
  if t['t']:
    t_taken = time.time() - t['t']
    print('Time taken', t_taken)
  t['t'] = time.time()

  if str:
    print(str)

def time_tracker(): return {'t': None}
