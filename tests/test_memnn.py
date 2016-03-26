from __future__ import print_function

import unittest
from seya.layers.memnn2 import MemN2N
from keras.models import Sequential
from keras.layers.core import Lambda
from keras import backend as K

import numpy as np


class TestMemNN(unittest.TestCase):
    """Test seya.layers.memnn layer"""

    def test_memnn(self):
        def identity_init(shape, name=None):
            dim = max(shape)
            I = np.identity(dim)[:shape[0], :shape[1]]
            return K.variable(I, name=name)

        input_dim = 20
        output_dim = 64
        input_length = 9
        memory_length = 7

        facts = Sequential()
        facts.add(Lambda(lambda x: x, input_shape=(memory_length, input_dim),
                         output_shape=(memory_length, input_dim)))
        question = Sequential()
        question.add(Lambda(lambda x: x, input_shape=(1, input_dim),
                            output_shape=(1, input_dim)))

        memnn = MemN2N([facts, question], output_dim, input_dim,
                       input_length, memory_length,
                       output_shape=(output_dim,))
        memnn.build()

        model = Sequential()
        model.add(memnn)
        model.compile("sgd", "mse")

        inp = np.random.randint(0, input_dim,
                                (1, memory_length, input_length))
        que = np.random.randint(0, input_dim, (1, 1, input_length))
        assert model.predict([inp, que]).shape == (1, output_dim)

if __name__ == '__main__':
    unittest.main(verbosity=2)
