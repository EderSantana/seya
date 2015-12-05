# encoding: utf-8
"""Test seya.layers.recurrent module"""

from __future__ import print_function

import unittest
import numpy as np

from keras.models import Sequential
from seya.layers.conv_rnn import ConvGRU


class TestConvGRU(unittest.TestCase):
    """Test seya.layers.conv_rnn layer"""

    def test_basic(self):
        """Just check that the ConvGRU layer can compile and run"""
        nb_samples, timesteps, ndim, filter_dim = 5, 10, 28, 3
        input_flat = ndim ** 2
        layer = ConvGRU(filter_dim=(1, filter_dim, filter_dim), reshape_dim=(1, ndim, ndim),
                        batch_size=nb_samples,
                        input_shape=(timesteps, input_flat),
                        return_sequences=True)
        model = Sequential()
        model.add(layer)
        model.compile('sgd', 'mse')

        x = np.random.randn(nb_samples, timesteps, input_flat)
        y = model.predict(x)
        assert y.shape == (nb_samples, timesteps, input_flat)


if __name__ == '__main__':
    unittest.main(verbosity=2)
