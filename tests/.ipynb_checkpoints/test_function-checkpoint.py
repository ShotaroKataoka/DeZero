import unittest

import numpy as np

from variable import Variable
from function import square, exp


def numerical_diff(f, x, eps=1e-4):
    x1 = Variable(np.array(x.data + eps))
    x2 = Variable(np.array(x.data - eps))
    y1 = f(x1).data
    y2 = f(x2).data
    return (y1-y2)/(eps*2)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        input = np.array(np.random.rand(1))
        x = Variable(input)
        y = square(x)
        expected = input**2
        self.assertEqual(y.data, expected)
        
    def test_backward(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
        
class ExpTest(unittest.TestCase):
    def test_forward(self):
        input = np.array(np.random.rand(1))
        x = Variable(input)
        y = exp(x)
        expected = np.exp(input)
        self.assertEqual(y.data, expected)
        
    def test_backward(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()
        num_grad = numerical_diff(exp, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
        
