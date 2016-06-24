import numpy as np
import theano
import theano.tensor as T
import keras.backend as K

from keras.layers.core import Layer

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

    def get_output_shape_for(self, input_shape):
        return (None, int(input_shape[1]),
                int(input_shape[2] / self.downsample_factor),
                int(input_shape[3] / self.downsample_factor))

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
        W[0, 0] = .9
        W[1, 1] = .9
        self.W = K.variable(W, name='{}_W'.format(self.name))
        M = np.ones((2, 3), dtype='float32')
        # M[0, 0] = 8.
        # M[1, 1] = 5.
        # M[1, 0] = 0.
        # M[0, 1] = 0.
        # M[0, 2] = 0.
        # M[1, 2] = 1.
        self.M = K.variable(M, name="{}_mask".format(self.name))
        self.trainable_weights = [self.W]

    def call(self, X, mask=None):
        theta = self.W * self.M
        theta = T.repeat(theta.dimshuffle('x', 0, 1), X.shape[0], axis=0)
        output = SpatialTransformer._transform(theta, X, self.downsample_factor)

        return output

    def output_shape_for(self, input_shape):
        return (None, input_shape[1],
                int(input_shape[2] / self.downsample_factor),
                int(input_shape[2] / self.downsample_factor))


class Cropper(Layer):
    """Homography layer
    """
    def __init__(self,
                 downsample_factor=1,
                 init_scale=1.,
                 ratio=1.,
                 **kwargs):
        self.downsample_factor = downsample_factor
        self.init_scale = init_scale
        self.ratio = ratio
        super(Cropper, self).__init__(**kwargs)

    def build(self, input_shape):
        W = np.zeros((4,), dtype='float32')
        W[0] = self.init_scale
        W[1] = self.init_scale
        self.W = K.variable(W, name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]

    def call(self, X, mask=None):
        sx = self.W[0:1]
        sy = self.W[1:2]
        tx = self.W[2:3]
        ty = self.W[3:]
        zero = K.zeros((1,))
        first_row = K.reshape(K.concatenate([sx, zero, tx]), (1, 3))
        second_row = K.reshape(K.concatenate([zero, sy, ty]), (1, 3))
        theta = K.concatenate([first_row, second_row], axis=0)
        theta = T.repeat(theta.dimshuffle('x', 0, 1), X.shape[0], axis=0)
        output = SpatialTransformer._transform(theta, X, self.downsample_factor)

        return output

    def output_shape_for(self, input_shape):
        return (None, input_shape[1],
                int(input_shape[2] / self.downsample_factor),
                int(input_shape[2] / self.downsample_factor))


class DifferentiableRAM(Layer):
    """DifferentiableRAM uses Gaussian attention mechanism from DRAW [1]_
    out_grid: list (height, width)
    References
    ----------
    """
    def __init__(self,
                 localization_net,
                 out_grid,
                 return_theta=False,
                 **kwargs):
        self.out_grid = out_grid
        self.locnet = localization_net
        self.return_theta = return_theta
        super(DifferentiableRAM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        self.regularizers = self.locnet.regularizers
        self.constraints = self.locnet.constraints
        self.width = input_shape[3]
        self.height = input_shape[2]

    def output_shape_for(self, input_shape):
        return (None, input_shape[1],
                int(self.out_grid[0]),
                int(self.out_grid[1]))

    def call(self, X, mask=None):
        p = self.locnet.call(X)
        gx, gy, sigma2, delta, gamma = self._get_attention_params(p)
        Fx, Fy = self._get_filterbank(gx, gy, sigma2, delta)
        output = self._read(X, gamma, Fx, Fy)
        if self.return_theta:
            return p
        else:
            return output

    def _get_attention_params(self, p):
        N = np.min(self.out_grid)
        gx = self.out_grid[0] * (p[:, 0]+1) / 2.
        gy = self.out_grid[1] * (p[:, 1]+1) / 2.
        sigma2 = T.exp(p[:, 2])
        delta = T.exp(p[:, 3]) * (max(self.width, self.height) - 1) / (N - 1.)
        gamma = T.exp(p[:, 4])
        return gx, gy, sigma2, delta, gamma

    def _get_filterbank(self, gx, gy, sigma2, delta):
        N = np.min(self.out_grid)
        small = 1e-4
        i1 = T.arange(self.out_grid[0]).astype("float32")
        i2 = T.arange(self.out_grid[1]).astype("float32")
        a = T.arange(self.width).astype("float32")
        b = T.arange(self.height).astype("float32")

        mx = gx[:, None] + delta[:, None] * (i1 - N/2. - .5)
        my = gy[:, None] + delta[:, None] * (i2 - N/2. - .5)

        Fx = T.exp(-(a - mx[:, :, None])**2 / 2. / sigma2[:, None, None])
        Fx /= (Fx.sum(axis=-1)[:, :, None] + small)
        Fy = K.exp(-(b - my[:, :, None])**2 / 2. / sigma2[:, None, None])
        Fy /= (Fy.sum(axis=-1)[:, :, None] + small)
        return Fx, Fy

    def _read(self, x, gamma, Fx, Fy):
        Fyx = (Fy[:, None, :, :, None] * x[:, :, None, :, :]).sum(axis=3)
        FxT = Fx.dimshuffle(0, 2, 1)
        FyxFx = (Fyx[:, :, :, :, None] * FxT[:, None, None, :, :]).sum(axis=3)
        return gamma[:, None, None, None] * FyxFx


class Translate(Layer):
    def __init__(self,
                 localization_net,
                 downsample_factor=1,
                 scale=[1., 1.],
                 **kwargs):
        self.downsample_factor = downsample_factor
        self.locnet = localization_net
        self.scale = scale
        self.return_theta = False
        super(Translate, self).__init__(**kwargs)

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
        vals = self.locnet.call(X)
        tx = vals[:, 4:5]
        ty = vals[:, 5:6]
        # sx = self.W[0:1]
        # sy = self.W[1:2]
        zero = K.zeros_like(tx)
        one = K.ones_like(tx)
        first_row = K.reshape(K.concatenate([one, zero, tx], axis=1), (-1, 1, 3))
        second_row = K.reshape(K.concatenate([zero, one, ty], axis=1), (-1, 1, 3))
        theta = K.concatenate([first_row, second_row], axis=1)
        theta = theta.reshape((X.shape[0], 2, 3))
        output = SpatialTransformer._transform(theta, X, self.downsample_factor)

        if self.return_theta:
            return theta.reshape((X.shape[0], 6))
        else:
            return output


class ProjectiveTransformer(Layer):
    """Projective Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    This implements the full 3x3 homography.

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
        return (input_shape[0], input_shape[1],
                int(input_shape[2] / self.downsample_factor),
                int(input_shape[3] / self.downsample_factor))

    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        theta = theta.reshape((X.shape[0], 3, 3))
        output = self._transform(theta, X, self.downsample_factor)

        if self.return_theta:
            return theta.reshape((X.shape[0], 9))
        else:
            return output

    @staticmethod
    def _transform(theta, input, downsample_factor):
        num_batch, num_channels, height, width = input.shape
        theta = theta.reshape((num_batch, 3, 3))  # T.reshape(theta, (-1, 2, 3))

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = T.cast(height, floatX)
        width_f = T.cast(width, floatX)
        out_height = T.cast(height_f // downsample_factor, 'int64')
        out_width = T.cast(width_f // downsample_factor, 'int64')
        grid = SpatialTransformer._meshgrid(out_height, out_width)

        # Transform A x (x_t, y_t, 1)^T -> (x_s / z_s, y_s / z_s)
        T_g = T.dot(theta, grid)
        x_s, y_s = T_g[:, 0] / T_g[:, 2], T_g[:, 1] / T_g[:, 2]
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
