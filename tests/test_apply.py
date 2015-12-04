# encoding: utf-8
"""Test seya.layers.recurrent module"""

from __future__ import print_function

import unittest
import numpy as np
import theano

from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers.core import Dense
from keras.models import Sequential

from seya.utils import apply_model
floatX = K.common._FLOATX


class TestApply(unittest.TestCase):
    """Test apply methods"""

    def test_apply_model(self):
        """Test keras.models.Sequential.__call__"""
        nb_samples, input_dim, output_dim = 3, 10, 5
        model = Sequential()
        model.add(Dense(output_dim=output_dim, input_dim=input_dim))
        model.compile('sgd', 'mse')

        X = K.placeholder(ndim=2)
        Y = apply_model(model, X)
        F = theano.function([X], Y)

        x = np.random.randn(nb_samples, input_dim).astype(floatX)
        y1 = F(x)
        y2 = model.predict(x)
        # results of __call__ should match model.predict
        assert_allclose(y1, y2)


if __name__ == '__main__':
    theano.config.exception_verbosity = 'high'
    unittest.main(verbosity=2)
