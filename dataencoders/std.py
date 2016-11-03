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

# Predicates
def func(xs):
  print(xs)
  return [x for x in xs if x]
non_empty = (lambda xs: [x for x in xs if x], None, identity, axis_apply)

# Other
lift_first_dim = (identity, None, identity, lambda _, arr: arr[0])
