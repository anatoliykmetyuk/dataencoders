import numpy as np
from dataencoders import *

def _test_words():
  words = np.array(['Foo', 'Bar', 'Char'], dtype=object)
  return process_array_by_elem(words, [lambda x: np.array([x] * 10, dtype=object).reshape(10)])


def _test_id_encoder():
  words = _test_words()
  print('Words', words)

  mapping = map_array(words)
  print('Mapping', mapping)

  encoder = IdEncoder(mapping)

  enc = encoder.encode(words)
  print('Encoded', enc)

  dec = encoder.decode(enc)
  print('Decoded', dec)

def _test_one_hot_encoder():
  words = _test_words()
  print('Words', words)

  mapping = map_array(words)
  id_enc  = IdEncoder(mapping)
  oh_enc  = OneHotEncoder()

  words_ids = id_enc.encode(words)
  print('Word ids', words_ids)

  words_oh = oh_enc.encode(words_ids)
  print('One-hot', words_oh)

  dec = oh_enc.decode(words_oh)
  print('Decoded from one-hot', dec)


_test_id_encoder()
_test_one_hot_encoder()
