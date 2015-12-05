from __future__ import division
import numpy as np

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import Recurrent
from keras.layers.core import MaskedLayer
from keras import initializations
from keras import activations
from keras import backend as K

from seya.utils import apply_layer, apply_model


class ConvGRU(Recurrent):
    def __init__(self, filter_dim, reshape_dim, batch_size, subsample=(1, 1),
                 init='glorot_uniform', inner_init='glorot_uniform',
                 activation='sigmoid', inner_activation='hard_sigmoid',
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

        super(ConvGRU, self).__init__(**kwargs)

    def build(self):
        batch_size = self.batch_size
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

        self.params = self.conv_h.params + self.conv_z.params + self.conv_r.params + \
            self.conv_x_h.params + self.conv_x_z.params + self.conv_x_r.params

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_initial_states(self, X):
        hidden_dim = np.prod(self.output_dim)
        h = K.zeros((self.batch_size, hidden_dim))
        return [h, ]

    def step(self, x, states):
        input_shape = (self.batch_size, ) + self.reshape_dim
        hidden_dim = (self.batch_size, ) + self.output_dim
        nb_filter, nb_rows, nb_cols = self.output_dim
        h_tm1 = K.reshape(states[0], hidden_dim)

        x_t = K.reshape(x, input_shape)
        xz_t = apply_layer(self.conv_x_z, x_t)
        xr_t = apply_layer(self.conv_x_r, x_t)
        xh_t = apply_layer(self.conv_x_h, x_t)

        xz_t = apply_layer(self.max_pool, xz_t)
        xr_t = apply_layer(self.max_pool, xr_t)
        xh_t = apply_layer(self.max_pool, xh_t)

        z = self.inner_activation(xz_t + apply_layer(self.conv_z, h_tm1))
        r = self.inner_activation(xr_t + apply_layer(self.conv_r, h_tm1))
        hh_t = self.activation(xh_t + apply_layer(self.conv_h, r * h_tm1))
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t = K.flatten(h_t)
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
    def __init__(self, model, batch_size, **kwargs):
        self.batch_size = batch_size
        self.model = model
        self.reshape_dim = (self.batch_size, ) + self.model.input_shape[1:]
        super(TimeDistributedModel, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        self.input = K.placeholder(shape=(self.batch_size, input_shape[1],
                                          input_shape[2]))
        self.model.build()
        self.parmas = self.model.params

    def _step(self, x_t, *args):
        x_t = K.reshape(x_t, self.reshape_dim)
        return apply_model(self.model, x_t)

    def get_output(self, train=False):
        X = self.get_input()
        batch_size, time_len = X.shape[:2]
        X = X.flatten(ndim=2)  # (sample*time, dim)
        X = K.reshape(X, self.reshape_dim)  # (sample*time, dim1, dim2, ...)
        Y = apply_model(self.model, X)
        Y = K.reshape(Y, (batch_size, time_len, -1))  # (sample, time, dim_out)
        return Y

    @property
    def output_shape(self):
        input_shape = self.input_shape
        output_shape = self.model.output_shape
        return input_shape[0], input_shape[1], np.prod(output_shape[1:])
