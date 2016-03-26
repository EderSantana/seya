from __future__ import division
import numpy as np
import theano.tensor as T

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import Recurrent
from keras.layers.core import MaskedLayer
from keras import initializations
from keras import activations
from keras import backend as K

from seya.utils import apply_layer


class ConvRNN(Recurrent):
    """RNN with all connections being convolutions:
    H_t = activation(conv(H_tm1, W_hh) + conv(X_t, W_ih) + b)
    with H_t and X_t being images and W being filters.

    We use Keras' RNN API, thus input and outputs should be 3-way tensors.
    Assuming that your input video have frames of size
    [nb_channels, nb_rows, nb_cols], the input of this layer should be reshaped
    to [batch_size, time_length, nb_channels*nb_rows*nb_cols]. Thus, you have to
    pass the original images shape to the ConvRNN layer.

    Parameters:
    -----------
    filter_dim: list [nb_filters, nb_row, nb_col] convolutional filter
        dimensions
    reshape_dim: list [nb_channels, nb_row, nb_col] original dimensions of a
        frame.
    batch_size: int, batch_size is useful for TensorFlow backend.
    time_length: int, optional for Theano, mandatory for TensorFlow
    subsample: (int, int), just keras.layers.Convolutional2D.subsample

    """
    def __init__(self, filter_dim, reshape_dim,
                 batch_size=None, subsample=(1, 1),
                 init='glorot_uniform', inner_init='glorot_uniform',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, **kwargs):
        self.batch_size = batch_size
        self.border_mode = 'same'
        self.filter_dim = filter_dim
        self.reshape_dim = reshape_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.initial_weights = weights

        self.subsample = tuple(subsample)
        self.output_dim = (filter_dim[0], reshape_dim[1]//self.subsample[0],
                           reshape_dim[2]//self.subsample[1])

        super(ConvRNN, self).__init__(**kwargs)

    def _get_batch_size(self, X):
        if K._BACKEND == 'theano':
            batch_size = X.shape[0]
        else:
            batch_size = self.batch_size
        return batch_size

    def build(self):
        if K._BACKEND == 'theano':
            batch_size = None
        else:
            batch_size = None  # self.batch_size
        input_dim = self.input_shape
        bm = self.border_mode
        reshape_dim = self.reshape_dim
        hidden_dim = self.output_dim

        nb_filter, nb_rows, nb_cols = self.filter_dim
        self.input = K.placeholder(shape=(batch_size, input_dim[1], input_dim[2]))

        # self.b_h = K.zeros((nb_filter,))
        self.conv_h = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)
        self.conv_x = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)

        # hidden to hidden connections
        self.conv_h.build()
        # input to hidden connections
        self.conv_x.build()

        self.max_pool = MaxPooling2D(pool_size=self.subsample, input_shape=hidden_dim)
        self.max_pool.build()

        self.trainable_weights = self.conv_h.trainable_weights + self.conv_x.trainable_weights

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_initial_states(self, X):
        batch_size = self._get_batch_size(X)
        hidden_dim = np.prod(self.output_dim)
        if K._BACKEND == 'theano':
            h = T.zeros((batch_size, hidden_dim))
        else:
            h = K.zeros((batch_size, hidden_dim))
        return [h, ]

    def step(self, x, states):
        batch_size = self._get_batch_size(x)
        input_shape = (batch_size, ) + self.reshape_dim
        hidden_dim = (batch_size, ) + self.output_dim
        nb_filter, nb_rows, nb_cols = self.output_dim
        h_tm1 = K.reshape(states[0], hidden_dim)

        x_t = K.reshape(x, input_shape)
        Wx_t = self.conv_x(x_t, train=True)
        h_t = self.activation(Wx_t + self.conv_h(h_tm1, train=True))
        h_t = K.batch_flatten(h_t)
        return h_t, [h_t, ]

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1], np.prod(self.output_dim))
        else:
            return (input_shape[0], np.prod(self.output_dim))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "filter_dim": self.filter_dim,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "return_sequences": self.return_sequences,
                  "reshape_dim": self.reshape_dim,
                  "go_backwards": self.go_backwards}
        base_config = super(ConvGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvGRU(ConvRNN):
    """ConvGRU is similar to ConvRNN, but with a GRU like state transition
    For documentation and details check seya.layers.conv_rnn.ConvRNN and
    keras.layers.recurrent.GRU

    """
    def __init__(self, filter_dim, reshape_dim, batch_size=None,
                 subsample=(1, 1),
                 init='glorot_uniform', inner_init='glorot_uniform',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, **kwargs):
        super(ConvGRU, self).__init__(
            filter_dim=filter_dim, reshape_dim=reshape_dim,
            batch_size=batch_size, subsample=subsample,
            init=init, inner_init=inner_init, activation=activation,
            inner_activation=inner_activation, weights=weights, **kwargs)

    def build(self):
        batch_size = None
        input_dim = self.input_shape
        bm = self.border_mode
        reshape_dim = self.reshape_dim
        hidden_dim = self.output_dim

        nb_filter, nb_rows, nb_cols = self.filter_dim
        self.input = K.placeholder(shape=(batch_size, input_dim[1], input_dim[2]))

        self.b_h = K.zeros((nb_filter,))
        self.b_r = K.zeros((nb_filter,))
        self.b_z = K.zeros((nb_filter,))

        self.conv_h = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)
        self.conv_z = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)
        self.conv_r = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)

        self.conv_x_h = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)
        self.conv_x_z = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)
        self.conv_x_r = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)

        # hidden to hidden connections
        self.conv_h.build()
        self.conv_z.build()
        self.conv_r.build()
        # input to hidden connections
        self.conv_x_h.build()
        self.conv_x_z.build()
        self.conv_x_r.build()

        self.max_pool = MaxPooling2D(pool_size=self.subsample)

        self.trainable_weights = self.conv_h.trainable_weights + self.conv_z.trainable_weights + self.conv_r.trainable_weights + \
            self.conv_x_h.trainable_weights + self.conv_x_z.trainable_weights + self.conv_x_r.trainable_weights

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        batch_size = self._get_batch_size(x)
        input_shape = (batch_size, ) + self.reshape_dim
        hidden_dim = (batch_size, ) + self.output_dim
        nb_filter, nb_rows, nb_cols = self.output_dim
        h_tm1 = K.reshape(states[0], hidden_dim)

        x_t = K.reshape(x, input_shape)
        xz_t = self.conv_x_z(x_t, train=True)
        xr_t = self.conv_x_r(x_t, train=True)
        xh_t = self.conv_x_h(x_t, train=True)

        xz_t = apply_layer(self.max_pool, xz_t)
        xr_t = apply_layer(self.max_pool, xr_t)
        xh_t = apply_layer(self.max_pool, xh_t)

        z = self.inner_activation(xz_t + self.conv_z(h_tm1))
        r = self.inner_activation(xr_t + self.conv_r(h_tm1))
        hh_t = self.activation(xh_t + self.conv_h(r * h_tm1))
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t = K.batch_flatten(h_t)
        return h_t, [h_t, ]

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1], np.prod(self.output_dim))
        else:
            return (input_shape[0], np.prod(self.output_dim))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "filter_dim": self.filter_dim,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "return_sequences": self.return_sequences,
                  "reshape_dim": self.reshape_dim,
                  "go_backwards": self.go_backwards}
        base_config = super(ConvGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeDistributedModel(MaskedLayer):
    """TimeDistributedModel
    Broadcast a model across the time dimensions

    Parameters:
    -----------
    model: a valid keras.models.Sequential
    batch_size: int, necessary for TensorFlow backend

    """
    def __init__(self, model, batch_size=None, **kwargs):
        self.model = model
        self.batch_size = batch_size
        if K._BACKEND == "tensorflow" and not batch_size:
            raise ValueError("Define batch_size")
        super(TimeDistributedModel, self).__init__(**kwargs)

    def _get_batch_size(self, X):
        if K._BACKEND == 'theano':
            batch_size = X.shape[0]
        else:
            batch_size = self.batch_size
        return batch_size

    def build(self):
        # if K._BACKEND == 'theano':
        #     batch_size = None
        # else:
        #     batch_size = self.batch_size
        input_shape = self.input_shape
        self.input = K.placeholder(shape=(None, input_shape[1],
                                          input_shape[2]))
        self.model.build()
        self.trainable_weights = self.model.trainable_weights

    def _step(self, x_t, *args):
        x_t = K.reshape(x_t, self.reshape_dim)
        x_t = self.model(x_t)
        return K.batch_flatten(x_t)

    # def get_output(self, train=False):
    #     X = self.get_input(train)

    #     def step(x, states):
    #         x = K.reshape(x, (-1, ) + self.model.input_shape[1:])
    #         output = self.model(x, train=train)
    #         return output, []

    #     last_output, outputs, states = K.rnn(step, X,
    #                                          initial_states=[],
    #                                          mask=None)
    #     return outputs

    def get_output(self, train=True):
        X = self.get_input(train)
        out_dim = np.prod(self.model.output_shape[1:])
        # batch_size = self._get_batch_size(X)
        if K._BACKEND == 'theano':
            time_len = K.shape(X)[1]
            new_shape = (-1, time_len, out_dim)
        else:
            # time_len = self.input_shape[2:3]
            time_len = K.shape(X)[1]
            new_shape = K.concatenate([np.asarray([-1, ]), time_len,
                                       np.asarray([out_dim, ])])

        reshape_dim = (-1, ) + self.model.input_shape[1:]
        Inp = K.batch_flatten(X)  # (sample*time, dim)
        Inp = K.reshape(Inp, reshape_dim)  # (sample*time, dim1, dim2, ...)
        Y = self.model(Inp, train=train)
        Y = K.reshape(Y, new_shape)  # (sample, time, dim_out)
        return Y

    @property
    def output_shape(self):
        input_shape = self.input_shape
        output_shape = self.model.output_shape
        return input_shape[0], input_shape[1], np.prod(output_shape[1:])
