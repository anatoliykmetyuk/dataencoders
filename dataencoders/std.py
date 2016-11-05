import numpy as np
from dataencoders import *

'''
Contains some useful primitives to use with this framework.
'''

class Padding(object):
  '''Represents a padding strategy'''

  def __init__(self, length, placeholder):
    self.length      = length
    self.placeholder = placeholder

  def __call__(self, arr):
    '''Apply this padding strategy to the given array.'''
    arr = arr[:self.length]
    return np.concatenate((arr, np.array([self.placeholder] * max(self.length - len(arr), 0))))

def string_splitter(at, pad): return lambda s: pad(np.array(s.split(at), dtype=object))

# Unfolders
def lines(pad): return string_splitter('\n', pad)
def words(pad): return string_splitter(' ' , pad)
def chars(pad): return lambda s: pad(list(s))

# Folders
def join_chars(chars): return ''.join(chars)
flatten_last_dim = (identity, None, identity, lambda _, arr: arr.reshape(arr.shape[:-2] + (arr.shape[-1] * arr.shape[-2],)))

# Predicates
def filter(predicate, pad):
  def f(xs):
    res = [x for x in xs if predicate(x)]
    res = pad(res)
    return res
  return (f, None, identity, axis_apply)
  
non_empty = filter(identity, identity)
def word_char(pad): return filter(lambda s: s.isalnum() or s.isspace(), pad)
def filter_indices(ids): return (lambda xs: xs[ids], None, identity, axis_apply)

# Maps
to_lowercase = lambda s: s.lower()

def flat_map(mapper): return lambda ax: np.concatenate([mapper(x) for x in ax], axis=0)

# Json
def json_array (name, pad=identity): return lambda x: pad(np.array(x[name], dtype=object))
def json_object(name): return lambda x: x[name]

# Other
lift_first_dim = (identity, None, identity, lambda _, arr: arr[0])
