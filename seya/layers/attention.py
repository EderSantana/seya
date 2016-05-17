import numpy as np
import theano
import theano.tensor as T
import keras.backend as K

from keras.layers.core import Layer

from ..utils import apply_model

floatX = theano.config.floatX


class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:

    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    """
    def __init__(self,
                 localization_net,
                 downsample_factor=1,
                 return_theta=False,
                 **kwargs):
        self.downsample_factor = downsample_factor
        self.locnet = localization_net
        self.return_theta = return_theta
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        self.regularizers = self.locnet.regularizers
        self.constraints = self.locnet.constraints

    def output_shape_for(self, input_shape):
        return (None, 3,
                int(input_shape[2] / self.downsample_factor),
                int(input_shape[2] / self.downsample_factor))


    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        theta = theta.reshape((X.shape[0], 2, 3))
        output = self._transform(theta, X, self.downsample_factor)

        if self.return_theta:
            return theta.reshape((X.shape[0], 6))
        else:
            return output

    @staticmethod
    def _repeat(x, n_repeats):
        rep = T.ones((n_repeats,), dtype='int32').dimshuffle('x', 0)
        x = T.dot(x.reshape((-1, 1)), rep)
        return x.flatten()

    @staticmethod
    def _interpolate(im, x, y, downsample_factor):
        # constants
        num_batch, height, width, channels = im.shape
        height_f = T.cast(height, floatX)
        width_f = T.cast(width, floatX)
        out_height = T.cast(height_f // downsample_factor, 'int64')
        out_width = T.cast(width_f // downsample_factor, 'int64')
        zero = T.zeros([], dtype='int64')
        max_y = T.cast(im.shape[1] - 1, 'int64')
        max_x = T.cast(im.shape[2] - 1, 'int64')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = T.cast(T.floor(x), 'int64')
        x1 = x0 + 1
        y0 = T.cast(T.floor(y), 'int64')
        y1 = y0 + 1

        x0 = T.clip(x0, zero, max_x)
        x1 = T.clip(x1, zero, max_x)
        y0 = T.clip(y0, zero, max_y)
        y1 = T.clip(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = SpatialTransformer._repeat(
            T.arange(num_batch, dtype='int32')*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat
        #  image and restore channels dim
        im_flat = im.reshape((-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # and finanly calculate interpolated values
        x0_f = T.cast(x0, floatX)
        x1_f = T.cast(x1, floatX)
        y0_f = T.cast(y0, floatX)
        y1_f = T.cast(y1, floatX)
        wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
        wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
        wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
        wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
        output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
        return output

    @staticmethod
    def _linspace(start, stop, num):
        # produces results identical to:
        # np.linspace(start, stop, num)
        start = T.cast(start, floatX)
        stop = T.cast(stop, floatX)
        num = T.cast(num, floatX)
        step = (stop-start)/(num-1)
        return T.arange(num, dtype=floatX)*step+start

    @staticmethod
    def _meshgrid(height, width):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = T.dot(T.ones((height, 1)),
                    SpatialTransformer._linspace(-1.0, 1.0, width).dimshuffle('x', 0))
        y_t = T.dot(SpatialTransformer._linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                    T.ones((1, width)))

        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))
        ones = T.ones_like(x_t_flat)
        grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    @staticmethod
    def _transform(theta, input, downsample_factor):
        num_batch, num_channels, height, width = input.shape
        theta = theta.reshape((num_batch, 2, 3))  # T.reshape(theta, (-1, 2, 3))

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = T.cast(height, floatX)
        width_f = T.cast(width, floatX)
        out_height = T.cast(height_f // downsample_factor, 'int64')
        out_width = T.cast(width_f // downsample_factor, 'int64')
        grid = SpatialTransformer._meshgrid(out_height, out_width)

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = T.dot(theta, grid)
        x_s, y_s = T_g[:, 0], T_g[:, 1]
        x_s_flat = x_s.flatten()
        y_s_flat = y_s.flatten()

        # dimshuffle input to  (bs, height, width, channels)
        input_dim = input.dimshuffle(0, 2, 3, 1)
        input_transformed = SpatialTransformer._interpolate(
            input_dim, x_s_flat, y_s_flat,
            downsample_factor)

        output = T.reshape(input_transformed,
                           (num_batch, out_height, out_width, num_channels))
        output = output.dimshuffle(0, 3, 1, 2)
        return output


class Homography(Layer):
    """Homography layer
    """
    def __init__(self,
                 downsample_factor=1,
                 **kwargs):
        self.downsample_factor = downsample_factor
        super(Homography, self).__init__(**kwargs)

    def build(self, input_shape):
        W = np.zeros((2, 3), dtype='float32')
        W[0, 0] = 1.
        W[1, 1] = 1.
        self.W = K.variable(W, name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
        self.regularizers()

    def call(self, X, mask=None):
        theta = self.W
        theta = T.repeat(self.W.dimshuffle('x', 0, 1), X.shape[0], axis=0)
        output = SpatialTransformer._transform(theta, X, self.downsample_factor)

        return output

    def output_shape_for(self, input_shape):
        return (None, input_shape[1],
                int(input_shape[2] / self.downsample_factor),
                int(input_shape[2] / self.downsample_factor))
