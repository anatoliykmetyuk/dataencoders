import time

def chain(inp, methods, benchmark=None):
  if benchmark: benchmark.reset()

  res = inp
  for m in methods:
    if benchmark: benchmark('Executing ' + m.__name__)
    res = m(res)

  if benchmark: benchmark.done()
  return res

class Benchmark(object):
  def __init__(self
    , current_time = time.time
    , time_taken_str = lambda t: 'Time taken: ' + str(t)
    , log = lambda msg: print(msg)
  ):
    self.previous_time = None
    
    self.current_time   = current_time
    self.time_taken_str = time_taken_str
    self.log            = log

  def __call__(self, str):
    if self.previous_time:
      t_taken = self.current_time() - self.previous_time
      self.log(self.time_taken_str(t_taken))
    self.previous_time = self.current_time()

    if str:
      self.log(str)

  def reset(self): self.previous_time = None

  def done(self): self(None)
