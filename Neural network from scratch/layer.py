# Base class
class Layer:
  def __init__(self):
    self.input = None
    self.output = None

  # Computes the output of Y of a layer for a given X
  def forward_propogation(self, input):
    raise NotImplementedError
  
  # Computes dE/dX for a given dE/dY (and update parameters if any)
  def backward_propogation(self, output_error, learning_rate):
    raise NotImplementedError
