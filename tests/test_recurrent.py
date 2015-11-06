# encoding: utf-8
"""Test seya.layers.recurrent module"""

from __future__ import print_function

import unittest
import numpy as np
import theano

from keras.layers.recurrent import SimpleRNN
from seya.layers.recurrent import Bidirectional


class TestBidirectional(unittest.TestCase):
    """Test seya.layers.recurrent.Bidirectional layer"""

    def test_basic(self):
        """Just check that the Bidirectional layer can compile and run"""
        nb_samples, timesteps, input_dim, output_dim = 3, 3, 10, 5

        for ret_seq in [True, False]:
            rnn1 = SimpleRNN(output_dim, return_sequences=ret_seq,
                             input_shape=(None, input_dim))
            rnn2 = SimpleRNN(output_dim, return_sequences=ret_seq,
                             input_shape=(None, input_dim))
            layer = Bidirectional(rnn1, rnn2, return_sequences=ret_seq)
            layer.input = theano.shared(value=np.ones((nb_samples, timesteps, input_dim)))
            rnn1.input = layer.input
            rnn2.input = layer.input
            _ = layer.get_config()

            for train in [True, False]:
                out = layer.get_output(train).eval()
                # Make sure the output has the desired shape
                if ret_seq:
                    assert(out.shape == (nb_samples, timesteps, output_dim*2))
                else:
                    assert(out.shape == (nb_samples, output_dim*2))
                _ = layer.get_output_mask(train)


if __name__ == '__main__':
    theano.config.exception_verbosity = 'high'
    unittest.main(verbosity=2)
