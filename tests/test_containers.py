import numpy as np
from numpy.testing import assert_allclose

import unittest
import theano.tensor as T
from theano import function
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from seya.layers.containers import Recursive
from seya.layers.base import Dense2, Lambda


class TestRecursive(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRecursive, self).__init__(*args, **kwargs)
        self.input_dim = 2
        self.state_dim = 2
        self.model = Recursive(return_sequences=True)
        self.model.add_input('input', ndim=3)  # Input is 3D tensor
        self.model.add_state('h', dim=self.state_dim)
        self.model.add_node(Dense(self.input_dim + self.state_dim, self.state_dim,
                                  init='one'), name='rec',
                            inputs=['input', 'h'],
                            return_state='h')
        self.model.add_node(Activation('linear'), name='out', input='rec',
                            create_output=True)

        self.model2 = Sequential()
        self.model2.add(SimpleRNN(input_dim=self.input_dim, activation='linear',
                                  inner_init='one',
                                  output_dim=self.state_dim, init='one',
                                  return_sequences=True))

    def test_step(self):
        XX = T.matrix()
        HH = T.matrix()
        A = self.model._step(XX, HH)
        F = function([XX, HH], A)
        x = np.ones((1, 2))
        h = np.ones((1, 2))
        y = F(x, h)
        r = np.asarray([[4., 4.]])
        assert_allclose([r, r], y)

    def test_get_get_output(self):
        X = self.model.get_input()
        Y = self.model._get_output()
        F = function([X], Y, allow_input_downcast=True)

        x = np.ones((3, 5, self.input_dim))
        y = F(x)
        print y

        X2 = self.model2.get_input()
        Y2 = self.model2.get_output()
        F2 = function([X2], Y2)
        y2 = F2(x)

        assert_allclose(y2, y[1])


class TestOrthoRNN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestOrthoRNN, self).__init__(*args, **kwargs)
        self.input_dim = 2
        self.state_dim = 2
        self.model = Recursive(return_sequences=True)
        self.model.add_input('input', ndim=3)  # Input is 3D tensor
        self.model.add_state('h', dim=self.state_dim)
        self.model.add_node(Dense(self.input_dim, self.state_dim,
                                  init='one'), name='i2h',
                            inputs=['input', ])
        self.model.add_node(Dense(self.state_dim, self.state_dim,
                                  init='orthogonal'), name='h2h',
                            inputs=['h', ])
        self.model.add_node(Lambda(lambda x: x), name='rec',
                            inputs=['i2h', 'h2h'], merge_mode='sum',
                            return_state='h',
                            create_output=True)

        self.model2 = Sequential()
        self.model2.add(SimpleRNN(input_dim=self.input_dim, activation='linear',
                                  inner_init='one',
                                  output_dim=self.state_dim, init='one',
                                  return_sequences=True))
        U = self.model.nodes['h2h'].W.get_value()
        self.model2.layers[0].U.set_value(U)

    def test_step(self):
        XX = T.matrix()
        HH = T.matrix()
        A = self.model._step(XX, HH)
        F = function([XX, HH], A, on_unused_input='warn')
        x = np.ones((1, 2))
        h = np.ones((1, 2))
        y = F(x, h)
        assert(y[-1].shape == (1, 2))

    def test_get_get_output(self):
        X = self.model.get_input()
        Y = self.model._get_output()
        F = function([X], Y, allow_input_downcast=True)

        x = np.ones((3, 5, self.input_dim))
        y = F(x)
        print y

        X2 = self.model2.get_input()
        Y2 = self.model2.get_output()
        F2 = function([X2], Y2)
        y2 = F2(x)

        assert_allclose(y2, y[-1])
if __name__ == '__main__':
    print('Test Recursive container')
    unittest.main()
