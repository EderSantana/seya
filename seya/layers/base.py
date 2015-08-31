import numpy as np
import theano
import theano.tensor as T

from itertools import combinations
from keras.layers.core import MaskedLayer, Layer, Dense
from keras.utils.theano_utils import ndim_tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX


class Lambda(MaskedLayer):
    def __init__(self, func, ndim=2):
        super(Lambda, self).__init__()
        self.input = ndim_tensor(ndim)
        self.func = func

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.func(X)


class Pass(MaskedLayer):
    ''' Do literally nothing
        It can the first layer
    '''
    def __init__(self, ndim=2):
        super(Pass, self).__init__()
        self.input = ndim_tensor(ndim)

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
    def __init__(self, avg=0., std=1.):
        super(GaussianProd, self).__init__()
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
    '''Replicates an input matrix across a new second dimension.
        Originally useful for broadcasting a fixed input into a scan loop.
        Think conditional RNNs without the need to rewrite the RNN class.
    '''
    def __init__(self, leng):
        super(Replicator, self).__init__()
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


class OrthogonalDense(Dense):
    '''Dense layer with weights fixed to be Orthogonal

    ... WORK IN PROGRESS ...

    '''
    def __init__(self, dim):
        super(OrthogonalDense, self).__init__(input_dim=dim,
                                              output_dim=dim)
        self.n_free = dim * (dim-1) / 2
        self.W = self.W.flatten()[:self.n_free]
        self.dim = dim
        self.params = [self.W, self.params[1]]

    def _get_rotation(self):
        A = T.eye(self.dim)
        for i, (x, y) in enumerate(combinations(range(self.dim), 2)):
            B = T.eye(self.dim)

            if x == 0:
                b0 = []
                v3 = []
            else:
                b0 = T.zeros((x,))
                v3 = T.zeros((x, self.dim))
            if self.dim-x-y-1 == 0:
                b1 = []
            else:
                b1 = T.zeros((self.dim-x-y-1,))
            if self.dim-y+x+1 == 0:
                v4 = []
            else:
                v4 = T.zeros((self.dim-y+x+1, self.dim))
            if y-1 == 0:
                b2 = []
            else:
                b2 = T.zeros((y-1,))

            v1 = T.concatenate([b0,
                               T.cos(self.W[i]),
                               b2,
                               -T.sin(self.W[i]),
                               b1])
            v2 = T.concatenate([b0,
                               T.sin(self.W[i]),
                               b2,
                               T.cos(self.W[i]),
                               b1])

            B = T.concatenate([v1, v2, v3, v4], axis=0) + T.eye(self.dim)
            A = T.dot(A, B)
        return A

    def get_output(self, train=False):
        X = self.get_input(train)
        A = self._get_rotation()
        return T.dot(X, A)


class Dense2(Dense):
    '''
        Just your regular fully connected NN layer.
    '''
    def __init__(self, *args, **kwargs):

        super(Dense2, self).__init__(*args, **kwargs)

    def _get_output(self, X):
        output = self.activation(T.dot(X, self.W) + self.b)
        return output
