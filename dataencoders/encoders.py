import numpy as np
from .nparray import *

'''
Defines a concpet of encoding cathegorical data into
different forms.
'''

class Encoder(object):
  def encode(self, arr):
    '''
    :param arr: numpy array of arbitrary dimensionality.
      Every element of it will be encoded by this Encoder's strategy.
    '''
    raise NotImplementedError

  def decode(self, arr):
    '''
    An inverse of `encode`. For every Encoder, arr === decode(encode(arr)).
    '''
    raise NotImplementedError

class IdEncoder(Encoder):
  
  def __init__(self, id_to_obj):
    '''
    :param objToInt: a dictionary `id -> object`.
    '''
    self.id_to_obj = id_to_obj
    self.obj_to_id = {obj: idx for idx, obj in id_to_obj.items()}

  def encode(self, arr):
    return process_array_by_elem(arr, [lambda obj: self.obj_to_id[obj]], wrap_type=np.int)

  def decode(self, arr):
    return process_array_by_elem(arr, [lambda idx: self.id_to_obj[idx]], wrap_type=object)

class OneHotEncoder(Encoder):

  def encode(self, arr):
    max_id = arr.flatten().max()
    embeddings = np.eye(max_id + 1)
    return process_array_by_elem(arr, [lambda idx: embeddings[idx]])

  def decode(self, arr):
    return process_array_by_axis(arr, [lambda ax: np.argmax(ax)], wrap_type=np.int)

class RealEncoder(Encoder):
  def __init__(self, lower, upper):
    self.lower = lower
    self.upper = upper

  def encode(self, arr): return process_array_by_elem(arr, [self.int_to_float], wrap_type=np.float)
  def decode(self, arr): return process_array_by_elem(arr, [self.float_to_int], wrap_type=np.int  )

  def int_to_float(self, x): return float(x - self.lower) / float(self.upper - self.lower)
  def float_to_int(self, x): return int(np.round((x * float(self.upper - self.lower) + self.lower)))

def map_array(arr, start=0):
  '''
  Creates a dictionary with int ids as keys and all the elements of
  `arr` as values.

  :param start: the smallest index at which the mapping starts.
  '''
  objs = list(set(arr.flatten()))
  return {i + start: obj for i, obj in enumerate(objs)}
