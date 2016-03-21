import numpy as np

from keras import backend as K
from keras.layers.core import Layer

from seya.regularizers import LambdaRegularizer


class ITLRegularizer(Layer):
    '''ITL regularization layer using shared variables

    Parameters:
    -----------
    scale: float, (default, 1) how important is this regularization to the cost
           function.
    distribution: function, random number generator from the prior distribution
    ksize: float, (default, 1) Parzen window size.

    '''
    def default_distribution(shape):
        return K.random_normal(shape, std=5)

    def __init__(self, scale=1., distribution=default_distribution, distance='euclidean', ksize=1., **kwargs):
        super(ITLRegularizer, self).__init__(**kwargs)
        self.scale = scale
        self.distribution = distribution
        self.ksize = ksize
        self.distance = distance

    def _get_kernel(self, X, Z):
        G = K.sum((K.expand_dims(X, dim=1) - Z)**2, axis=-1)  # Gram matrix
        G = K.exp(-G/(self.ksize)) / K.sqrt(2*np.pi*self.ksize)
        return G

    def build(self):
        X = self.get_input()
        # if self.distribution == 'normal':
        #    Z = K.random_normal(K.shape(X), std=5)
        Z = self.distribution(K.shape(X))

        Gxx = self._get_kernel(X, X)
        Gzz = self._get_kernel(Z, Z)
        Gxz = self._get_kernel(X, Z)
        if self.distance == 'euclidean':  # same as minimum mean discrepancy
            itl_divergence = K.mean(Gxx) + K.mean(Gzz) - 2*K.mean(Gxz)

        elif self.distance == 'csd':  # Cauchy-Schwarz divergence
            itl_divergence = K.log(K.sqrt(K.mean(Gxx)*K.mean(Gzz)) /
                                   K.mean(Gxz))

        elif self.distance == 'entropy':  # Maximum entropy
            itl_divergence = K.mean(Gxx)

        elif self.distance == 'cip':  # cross-information potential
            itl_divergence = -K.mean(Gxz)

        self.regularizers = [LambdaRegularizer(self.scale * itl_divergence), ]


class ITLextInput(ITLRegularizer):
    """
    Same as ITLRegularizer but expecting random noise as an external input.
    This is meant to be used as part of a models.Graph
    """
    def __init__(self, code_size, scale=1., distance='euclidean', ksize=1., **kwargs):
        super(ITLextInput, self).__init__(scale=scale, distance=distance,
                                          ksize=ksize, **kwargs)
        self.code_size = code_size

    def build(self):
        Inp = self.get_input()
        X = Inp[:, :self.code_size]
        Z = Inp[:, self.code_size:]

        Gxx = self._get_kernel(X, X)
        Gzz = self._get_kernel(Z, Z)
        Gxz = self._get_kernel(X, Z)
        if self.distance == 'euclidean':  # same as minimum mean discrepancy
            itl_divergence = K.mean(Gxx) + K.mean(Gzz) - 2*K.mean(Gxz)

        elif self.distance == 'csd':  # Cauchy-Schwarz divergence
            itl_divergence = K.log(K.sqrt(K.mean(Gxx)*K.mean(Gzz)) /
                                   K.mean(Gxz))

        elif self.distance == 'entropy':  # Maximum entropy
            itl_divergence = K.mean(Gxx)

        self.regularizers = [LambdaRegularizer(self.scale * itl_divergence), ]

    def get_output(self, train=False):
        Inp = self.get_input(train)
        X = Inp[:, :self.code_size]
        Z = Inp[:, self.code_size:]
        Y = X
        return Y + 0*Z

    @property
    def output_shape(self):
        return (None, self.code_size)
