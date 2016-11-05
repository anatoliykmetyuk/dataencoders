import numpy as np
import os
import pickle

def loadLines(fname):
  with open(fname) as f:
    content = f.readlines()
    return np.array(content, dtype=object)

def load(fname):
  with open(fname) as f:
    return f.read()

def save_model(paths, data, storages):
  for p, d, s in zip(paths, data, storages):
    s.serialize(p, d)

def load_model(paths, storages):
  return [s.deserialize(p) for p, s in zip(paths, storages)]


# Serializers and deserializers
class Storage(object):
  def serialize  (self, path, data): raise NotImplementedError
  def deserialize(self, path      ): raise NotImplementedError

class FileStorage(Storage):
  def __init__(self, dir='.'):
    self.dir=dir
  
  def in_data_dir(self, path): return os.path.join(self.dir, path)


class NpStorage(FileStorage):
  def serialize(self, path, data):
    with open(self.in_data_dir(path), 'wb') as f: np.save(f, data)
  
  def deserialize(self, path):
    with open(self.in_data_dir(path), 'rb') as f: return np.load(f)

class PickleBinaryStorage(FileStorage):
  def serialize  (self, path, data):
    with open(self.in_data_dir(path), 'wb') as f: pickle.dump(data, f)
  
  def deserialize(self, path):
    with open(self.in_data_dir(path), 'rb') as f: return pickle.load(f)
