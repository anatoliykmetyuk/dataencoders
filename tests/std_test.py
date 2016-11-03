import numpy as np
from dataencoders import std
from dataencoders import *

def _test_string(): return '''Stack overflow

So what i want to do is essentially suck a line of txt from a .txt file, then assign the characters to a list, and then creat a list of all the separate characters in a list.

So a list of lists.

At the moment, I've tried:'''


def _test_splitters():
  target = np.array([_test_string()], dtype=object)
  print('Input:\n', target.shape, target)

  char_model = process_array_by_elem(target, [
    std.lines(identity)
  , std.non_empty
  , std.words(std.Padding(5 , ''))
  , std.chars(std.Padding(10, ''))
  , std.lift_first_dim
  ])

  print(char_model, char_model.shape, char_model.dtype)

_test_splitters()
