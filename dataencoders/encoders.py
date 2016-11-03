import numpy as np
from nparray import *

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

def map_array(arr, start=0):
  '''
  Creates a dictionary with int ids as keys and all the elements of
  `arr` as values.

  :param start: the smallest index at which the mapping starts.
  '''
  objs = list(set(arr.flatten()))
  return {i + start: obj for i, obj in enumerate(objs)}


### TESTS ###

def _test_map_array():
  words = np.array(['Foo', 'Bar', 'Char'], dtype=object)
  words = process_array_by_elem(words, [lambda x: np.array([x] * 10, dtype=object).reshape(10)])
  print('Words', words)

  mapping = map_array(words)
  print('Mapping', mapping)

  encoder = IdEncoder(mapping)

  enc = encoder.encode(words)
  print('Encoded', enc)

  dec = encoder.decode(enc)
  print('Decoded', dec)

_test_map_array()
