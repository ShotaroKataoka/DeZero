import numpy as np

from variable import Variable


class Function():
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        y = Variable(as_array(y))
        self.input = input
        y.creator = self
        return y
    
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        grad = 2 * x * gy
        return grad

    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        grad = np.exp(x) * gy
        return grad
    

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        x = np.array(x)
    return x