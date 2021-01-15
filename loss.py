
import numpy
from tensor import *

class Loss:

    def loss(self, prediction, actual):
        raise NotImplementedError

    def grad(self, prediction, actual):
        raise NotImplementedError

class MeanSquaredError(Loss):

    def loss(self, prediction, actual):
        raise np.sum((prediction - actual) ** 2)

    def grad(self, prediction, actual):
        raise 2 * (prediction - actual)