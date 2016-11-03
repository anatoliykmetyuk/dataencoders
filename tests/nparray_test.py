import numpy as np
from dataencoders import *

def _test_process_array_by_axis():
  arr = np.arange(10)
  res = process_array_by_axis(arr, [
    lambda ax: ax + 2
  , lambda ax: ax * 2
  ])

  print(res)

def _test_process_array_by_axis_fold():
  arr = np.arange(10)
  res = process_array_by_axis(arr, [
    lambda ax: np.sum(ax)
  ], wrap_type=np.int)

  print(res)

def _test_process_array_by_elem():
  arr = np.arange(10)
  res = process_array_by_elem(arr, [
    lambda x: np.arange(10) * x
  ])

  print(res)

def _test_process_array_by_elem_scalar():
  arr = np.arange(10)
  res = process_array_by_elem(arr, [
    lambda x: x + 1
  , lambda x: x * 2
  ])

  print(res)

def _test_custom_wrapper_types():
  arr = np.arange(10)
  res = process_array_by_elem(arr, [
    (lambda x: 'a' * x, object)
  ])

  print(res)

_test_process_array_by_axis()
_test_process_array_by_axis_fold()
_test_process_array_by_elem()
_test_process_array_by_elem_scalar()
_test_custom_wrapper_types()
