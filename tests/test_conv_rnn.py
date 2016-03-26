# encoding: utf-8
"""Test seya.layers.recurrent module"""

from __future__ import print_function

import unittest
import numpy as np

from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
from keras.layers.convolutional import Convolution2D
from seya.layers.conv_rnn import ConvRNN, ConvGRU, TimeDistributedModel


class TestConvRNNs(unittest.TestCase):
    """Test seya.layers.conv_rnn layer"""

    def test_conv_rnn(self):
        """Just check that the ConvRNN layer can compile and run"""
        nb_samples, timesteps, ndim, filter_dim = 5, 10, 28, 3
        input_flat = ndim ** 2
        layer = ConvRNN(filter_dim=(1, filter_dim, filter_dim),
                        reshape_dim=(1, ndim, ndim),
                        input_shape=(timesteps, input_flat),
                        return_sequences=True)
        model = Sequential()
        model.add(layer)
        model.add(TimeDistributedDense(10))
        model.compile('sgd', 'mse')

        x = np.random.randn(nb_samples, timesteps, input_flat)
        y = model.predict(x)
        assert y.shape == (nb_samples, timesteps, 10)

    def test_conv_gru(self):
        """Just check that the ConvGRU layer can compile and run"""
        nb_samples, timesteps, ndim, filter_dim = 5, 10, 28, 3
        input_flat = ndim ** 2
        layer = ConvGRU(filter_dim=(1, filter_dim, filter_dim),
                        reshape_dim=(1, ndim, ndim),
                        # input_shape=(timesteps, input_flat),
                        return_sequences=True)
        model = Sequential()
        model.add(TimeDistributedDense(input_flat, input_dim=input_flat))
        model.add(layer)
        model.compile('sgd', 'mse')

        x = np.random.randn(nb_samples, timesteps, input_flat)
        y = model.predict(x)
        assert y.shape == (nb_samples, timesteps, input_flat)

    def test_time_distributed(self):
        """Just check that the TimeDistributedModel layer can compile and run"""
        nb_samples, timesteps, ndim, filter_dim = 5, 10, 28, 3
        input_flat = ndim ** 2

        inner = Sequential()
        inner.add(Convolution2D(1, filter_dim, filter_dim, border_mode='same',
                                input_shape=(1, ndim, ndim)))

        layer = TimeDistributedModel(
            inner, batch_size=nb_samples, input_shape=(timesteps, input_flat))
        model = Sequential()
        model.add(layer)
        model.add(TimeDistributedDense(10))
        model.compile('sgd', 'mse')

        x = np.random.randn(nb_samples, timesteps, input_flat)
        y = model.predict(x)
        assert y.shape == (nb_samples, timesteps, 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
