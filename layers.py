
from typing import Callable, Dict
from tensor import *
import numpy as np

class Layer:

    def __init__(self):
        self.params = {}
        self.gradients = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError
   
F = Callable[[Tensor], Tensor]

class LinearLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, gradients):
        self.gradients['b'] = np.sum(gradients, axis=0)
        self.gradients['w'] = self.inputs.T @ gradients
        return gradients @ self.params['w'].T

class ActivationLayer(Layer):
    
    def __init__(self, function, function_derivitive):
        super().__init__()
        self.function = function
        self.function_derivitive = function_derivitive

    def tanh(self, inputs):
        return np.tanh(inputs)

    def tanh_derivitive(self, inputs):
        y = np.tanh(inputs)
        return 1 - y ** 2