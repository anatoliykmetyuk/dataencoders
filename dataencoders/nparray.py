import numpy as np

'''
Defines a pipeline to convert given data to Numpy array.
'''

def process_array(arr, pipe, wrap_type, pipe_adapter, apply_pipe):
  '''
  :param arr: raw array to convert
  :param pipe: functions to use for conversions on each stage.
  :param wrap_type: whether to wrap the result of the functions in the `pipe` into an array or not.
    Can be `None` to indicate that the functions already return arrays, or can be set to the type
    of the array the results will be wrapped in. Useful when the functions return scalar results.
  '''
  for f in pipe:
    if wrap_type:
      g = lambda x: np.array([pipe_adapter(f)(x)], dtype=wrap_type)
    else:
      g = pipe_adapter(f)

    arr = apply_pipe(g, arr)
    if arr.shape[-1] is 1:
      arr = arr.reshape(arr.shape[:-1])
  return arr


def process_array_by_axis(arr, pipe, wrap_type=None): return process_array(
  arr
, pipe
, pipe_adapter = lambda f: f
, apply_pipe   = lambda f, arr: np.apply_along_axis(f, len(arr.shape) - 1, arr)
, wrap_type    = wrap_type)

def process_array_by_elem(arr, pipe, wrap_type=None): return process_array(
  arr
, pipe
, pipe_adapter = lambda f: lambda x: f(x[0])
, apply_pipe   = lambda f, arr: np.apply_along_axis(f, len(arr.shape), arr.reshape(arr.shape + (1,)))
, wrap_type    = wrap_type
)
