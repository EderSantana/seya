import numpy as np
import theano
import theano.tensor as T

from keras.layers.core import MaskedLayer, Layer
from keras import backend as K
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX


class WinnerTakeAll2D(Layer):
    def __init__(self, **kwargs):
        super(WinnerTakeAll2D, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if train:
            M = K.max(X, axis=(2, 3), keepdims=True)
            R = K.switch(K.equal(X, M), X, 0.)
            return R
        else:
            return X


class Lambda(MaskedLayer):
    def __init__(self, func, output_shape, ndim=2, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=ndim)
        self.func = func
        self._output_shape = output_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.func(X)

    @property
    def output_shape(self):
        return self._output_shape


class Pass(MaskedLayer):
    ''' Do literally nothing
        It can the first layer
    '''
    def __init__(self, ndim=2, **kwargs):
        super(Pass, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=ndim)

    def get_output(self, train=False):
        X = self.get_input(train)
        return X


class GaussianProd(MaskedLayer):
    '''
        Multiply by Gaussian noise.
        Similar to dropout but with gaussians instead of binomials.
        The way they have this at Keras is not the way we need for
        Variational AutoEncoders.
    '''
    def __init__(self, avg=0., std=1., **kwargs):
        super(GaussianProd, self).__init__(**kwargs)
        self.std = std
        self.avg = avg
        self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train=False):
        X = self.get_input(train)
        X *= self.srng.normal(size=X.shape,
                              avg=self.avg,
                              std=self.std,
                              dtype=floatX)
        return X

    def get_config(self):
        return {"name": self.__class__.__name__,
                "avg": self.avg,
                "std": self.std}


class Replicator(MaskedLayer):
    '''
        WARN: use `keras.layer.RepeatVector` instead.

        Replicates an input matrix across a new second dimension.
        Originally useful for broadcasting a fixed input into a scan loop.
        Think conditional RNNs without the need to rewrite the RNN class.
    '''
    def __init__(self, leng):
        super(Replicator, self).__init__()
        raise ValueError("Deprecated. Use `keras.layers.RepeatVector instead`")
        self.ones = T.ones((leng,))
        self.input = T.matrix()

    def get_output(self, train=False):
        X = self.get_input(train)
        output = X[:, None, :] * self.ones[None, :, None]
        return output


class Unpool(Layer):
    '''Unpooling layer for convolutional autoencoders
    inspired by: https://github.com/mikesj-public/convolutional_autoencoder/blob/master/mnist_conv_autoencode.py

    Parameter:
    ----------
    ds: list with two values each one defines how much that dimension will
    be upsampled.
    '''
    def __init__(self, ds):
        raise ValueError("Deprecated. Use `keras.layers.convolutional.UpSample instead`")
        super(Unpool, self).__init__()
        self.input = T.tensor4()
        self.ds = ds

    def get_output(self, train=False):
        X = self.get_input(train)
        output = X.repeat(self.ds[0], axis=2).repeat(self.ds[1], axis=3)
        return output


class TimePicker(MaskedLayer):
    def __init__(self, time=-1):
        '''Picks a single value in time from a recurrent layer
           without forgeting its input mask'''
        super(TimePicker, self).__init__()
        self.time = time
        self.input = T.tensor3()

    def get_output(self, train=False):
        X = self.get_input(train)
        return X[:, self.time, :]

    @property
    def output_shape(self):
        return self.input_shape[0, 2]
