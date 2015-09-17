import numpy as np
import theano
import theano.tensor as T

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
        super(SpatialTransformer, self).__init__()
        self.downsample_factor = downsample_factor
        self.locnet = localization_net
        self.params = localization_net.params
        self.regularizers = localization_net.regularizers
        self.constraints = localization_net.constraints
        self.input = self.locnet.input  # This must be T.tensor4()
        self.return_theta = return_theta

    def get_output(self, train=False):
        X = self.get_input(train)
        theta = apply_model(self.locnet, X)
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


class AttentionST(SpatialTransformer):
    '''
    A Spatial Transformer limitted to scaling,
    cropping and translation.
    '''
    def __init__(self, *args, **kwargs):
        super(AttentionST, self).__init__(*args, **kwargs)

    def get_output(self, train=False):
        X = self.get_input()
        # locnet.get_output(X) should be shape (batchsize, 6)
        mask = np.ones((2, 3))
        mask[1, 0] = 0
        mask[0, 1] = 0
        mask = theano.shared(mask.astype(floatX))
        theta = self.locnet.get_output(X).reshape((X.shape[0], 2, 3))
        theta = theta * mask[None, :, :]

        output = self._transform(theta, X, self.downsample_factor)
        if self.return_theta:
            return theta.reshape((X.shape[0], 6))
        else:
            return output


class ST2(Layer):
    '''This implementation is similar to the equations in the paper
    but uses a lot more memory
    '''
    def __init__(self,
                 localization_net,
                 img_shape,
                 downsample_factor=(1, 1),
                 return_theta=False,
                 **kwargs):
        super(ST2, self).__init__()
        self.ds = downsample_factor
        self.locnet = localization_net
        self.img_shape = img_shape
        self.params = localization_net.params
        self.regularizers = localization_net.regularizers
        self.constraints = localization_net.constraints
        self.input = localization_net.input  # this should be T.tensor4()
        self.return_theta = return_theta

    def get_output(self, train=False):
        X = self.get_input()
        # locnet.get_output(X) should be shape (batchsize, 6)
        theta = self.locnet.get_output(train)  # .reshape((X.shape[0], 2, 3))
        thetas = T.nnet.sigmoid(theta[:, :4])
        thetat = T.nnet.sigmoid(theta[:, 4:]) * X.shape[2]
        theta = T.concatenate([thetas.reshape((X.shape[0], 2, 2)),
                               thetat.reshape((X.shape[0], 2, 1))],
                              axis=2)

        output = self._transform(X, theta, self.ds)
        if self.return_theta:
            return theta.reshape((X.shape[0], 6))
        else:
            return output

    def _meshgrid(self, row, col):
        x, y = np.meshgrid(np.linspace(0, row-1, row),
                           np.linspace(0, col-1, col))
        # x, y = np.meshgrid(np.linspace(-1, 1, row),
        #                   np.linspace(-1, 1, col))
        X = theano.shared(x.astype(floatX))
        Y = theano.shared(y.astype(floatX))
        ones = T.ones_like(X)
        grid = T.concatenate([X[None, :, :], Y[None, :, :], ones[None, :, :]],
                             axis=0)
        return grid

    def _transform(self, X, theta, ds):
        b = X.shape[0]
        chan, row, col = self.img_shape
        new_row = row / ds[0]
        new_col = col / ds[1]
        grid = self._meshgrid(new_row, new_col)
        new_grid = T.tensordot(theta, grid.reshape((3, new_row*new_col)),
                               axes=(2, 0))
        output = []
        for i in range(chan):
            out = X[:, i, :, :, None] * T.maximum(
                0, 1 - abs(new_grid[:, None, None, 0, :] -
                           grid[None, 0, :, :, None])) * T.maximum(
                               0, 1 - abs(new_grid[:, None, None, 1, :]
                                          - grid[None, 1, :, :, None]))
            out = out.sum(axis=(1, 2)).reshape((b, row, col))
            output.append(out.reshape((b, new_row,
                                       new_col)).dimshuffle(0, 'x', 1, 2))
        output = T.concatenate(output, axis=1)
        return output
