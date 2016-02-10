# encoding: utf-8
"""Test seya.layers.recurrent module"""

from __future__ import print_function

import unittest
import numpy as np
import theano

from seya.layers.variational import VariationalDense
from keras import backend as K
floatX = K.common._FLOATX


class TestVariational(unittest.TestCase):
    """Test seya.layers.variational.VariationalDense layer"""

    def test_basic(self):
        """Just check that the Variational layer can compile and run"""
        nb_samples, input_dim, output_dim = 3, 10, 5
        layer = VariationalDense(input_dim=input_dim, output_dim=output_dim,
                                 batch_size=nb_samples)
        X = layer.get_input()
        Y1 = layer.get_output(train=True)
        Y2 = layer.get_output(train=False)
        F = theano.function([X], [Y1, Y2])

        y1, y2 = F(np.random.randn(nb_samples, input_dim).astype(floatX))
        assert y1.shape == (nb_samples, output_dim)
        assert y2.shape == (nb_samples, output_dim)


if __name__ == '__main__':
    theano.config.exception_verbosity = 'high'
    unittest.main(verbosity=2)
