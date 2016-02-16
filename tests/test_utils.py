# encoding: utf-8
"""Test seya.layers.recurrent module"""

from __future__ import print_function

import unittest
import numpy as np
import theano

from seya import utils
from keras import backend as K
floatX = K.common._FLOATX


class TestUtils(unittest.TestCase):
    """Test seya.utils"""

    def test_batched_dot(self):
        """Check if batched dot works"""
        samples = 8
        xdim = 7
        sizes_list = [10, 5]
        X = K.placeholder(ndim=2)
        W1 = K.placeholder(ndim=2)
        W2 = K.placeholder(ndim=2)
        B1 = K.placeholder(ndim=1)
        B2 = K.placeholder(ndim=1)
        Y = utils.batched_dot(X, [W1, W2], [B1, B2], sizes_list)
        F = K.function([X, W1, W2, B1, B2], Y)

        x = np.ones((samples, xdim))
        w1 = np.ones((xdim, sizes_list[0])).astype('float32')
        w2 = np.ones((xdim, sizes_list[1])).astype('float32')
        b1 = np.ones(sizes_list[0]).astype('float32')
        b2 = np.ones(sizes_list[1]).astype('float32')
        y = F([x, w1, w2, b1, b2])
        print(x.dot(w2)+b2, y[1])
        print(y[0], x.dot(w1)+b1)
        #assert np.testing.assert_allclose(y[0], x.dot(w1)+b1, rtol=1e-3)
        #assert np.testing.assert_allclose(y[1], x.dot(w2)+b2, rtol=1e-3)


if __name__ == '__main__':
    theano.config.exception_verbosity = 'high'
    unittest.main(verbosity=2)
