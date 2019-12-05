import numpy as np

class Variable():
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")
        self.data = data
        self.grad = None
        self.creator = None
        
    def backward(self):
        assert self.creator is not None
        self.grad = np.ones_like(self.data)
        y = self
        while True:
            if y.creator is None:
                break
            else:
                f = y.creator
            f.input.grad = f.backward(y.grad)
            y = f.input
        return y