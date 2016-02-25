"""Note: this code was modified from:

https://github.com/lpigou/Theano-3D-ConvNet/blob/master/LICENSE
by @lpigou and collaborators
"""
import numpy as np
import theano.tensor as T
import keras.backend as K
from keras.layers.core import Layer


class NormLayer(Layer):
    """ Normalization layer """

    def __init__(self, method="lcn", kernel_size=9, threshold=1e-4,
                 nb_channels=3,
                 use_divisor=True, **kwargs):
        """
        method: "lcn", "gcn", "mean"
        LCN: local contrast normalization
            kwargs:
                kernel_size=9, threshold=1e-4, use_divisor=True
        GCN: global contrast normalization
            kwargs:
                scale=1., subtract_mean=True, use_std=False, sqrt_bias=0.,
                min_divisor=1e-8
        MEAN: local mean subtraction
            kwargs:
                kernel_size=5
        """

        super(NormLayer, self).__init__(**kwargs)
        self.method = method
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_divisor = use_divisor
        self.nb_channels = nb_channels
        self.input = K.placeholder(ndim=4)

    def get_output(self, train=False):
        X = self.get_input()
        out = []
        if self.method == "lcn":
            for i in range(self.nb_channels):
                XX = X[:, i:i+1, :, :]
                out += [self.lecun_lcn(XX, self.kernel_size, self.threshold,
                                       self.use_divisor)]
            out = K.concatenate(out, axis=1)
        elif self.method == "gcn":
            out = self.global_contrast_normalize(X)
        elif self.method == "mean":
            out = self.local_mean_subtraction(X, self.kernel_size)
        else:
            raise NotImplementedError()
        return out

    def lecun_lcn(self, X, kernel_size=7, threshold=1e-4, use_divisor=True):
        """
        Yann LeCun's local contrast normalization
        Orginal code in Theano by: Guillaume Desjardins
        """

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = self.gaussian_filter(
            kernel_size).reshape(filter_shape)
        # filters = shared(_asarray(filters, dtype=floatX), borrow=True)
        filters = K.variable(filters)

        convout = K.conv2d(X, filters, filter_shape=filter_shape,
                           border_mode='same')

        # For each pixel, remove mean of kernel_sizexkernel_size neighborhood
        new_X = X - convout

        if use_divisor:
            # Scale down norm of kernel_sizexkernel_size patch
            sum_sqr_XX = K.conv2d(K.pow(K.abs(new_X), 2), filters,
                                  filter_shape=filter_shape, border_mode='same')

            denom = T.sqrt(sum_sqr_XX)
            per_img_mean = denom.mean(axis=[2, 3])
            divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
            divisor = T.maximum(divisor, threshold)

            new_X /= divisor

        return new_X

    def local_mean_subtraction(self, X, kernel_size=5):

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = self.mean_filter(kernel_size).reshape(filter_shape)
        filters = K.variable(filters)

        mean = K.conv2d(X, filters, filter_shape=filter_shape,
                        border_mode='same')
        return X - mean

    def global_contrast_normalize(self, X, scale=1., subtract_mean=True,
                                  use_std=False, sqrt_bias=0., min_divisor=1e-6):

        ndim = X.ndim
        if ndim not in [3, 4]:
            raise NotImplementedError("X.dim>4 or X.ndim<3")

        scale = float(scale)
        mean = X.mean(axis=ndim-1)
        new_X = X.copy()

        if subtract_mean:
            if ndim == 3:
                new_X = X - mean[:, :, None]
            else:
                new_X = X - mean[:, :, :, None]

        if use_std:
            normalizers = T.sqrt(sqrt_bias + X.var(axis=ndim-1)) / scale
        else:
            normalizers = T.sqrt(sqrt_bias + (new_X ** 2).sum(axis=ndim-1)) / scale

        # Don't normalize by anything too small.
        T.set_subtensor(normalizers[(normalizers < min_divisor).nonzero()], 1.)

        if ndim == 3:
            new_X /= (normalizers[:, :, None] + 1e-6)
        else:
            new_X /= (normalizers[:, :, :, None] + 1e-6)

        return new_X

    def gaussian_filter(self, kernel_shape):

        x = np.zeros((kernel_shape, kernel_shape), dtype='float32')

        def gauss(x, y, sigma=2.0):
            Z = 2 * np.pi * sigma**2
            return 1./Z * np.exp(-(x**2 + y**2) / (2. * sigma**2))
        mid = np.floor(kernel_shape / 2.)
        for i in xrange(0, kernel_shape):
            for j in xrange(0, kernel_shape):
                x[i, j] = gauss(i-mid, j-mid)
        return x / sum(x)

    def mean_filter(self, kernel_size):
        s = kernel_size**2
        x = np.repeat(1./s, s).reshape((kernel_size, kernel_size))
        return x
