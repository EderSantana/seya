import theano
import theano.tensor as T

from keras.layers.core import Layer
from keras import activations, initializations
from keras.utils.theano_utils import alloc_zeros_matrix
from keras.regularizers import l2

from ..utils import diff_abs


def _RMSPropStep(cost, states, accum_1, accum_2):
    rho = .9
    lr = .001
    momentum = .9
    epsilon = 1e-8

    grads = T.grad(cost, states)

    new_accum_1 = rho * accum_1 + (1 - rho) * grads**2
    new_accum_2 = momentum * accum_2 - lr * grads / T.sqrt(new_accum_1 + epsilon)
    new_states = states + momentum * new_accum_2 - lr * (grads /
                                                         T.sqrt(new_accum_1 + epsilon))
    return new_states, new_accum_1, new_accum_2


class SparseCoding(Layer):
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform',
                 activation='linear',
                 truncate_gradient=-1,
                 gamma=.1,
                 n_steps=10,
                 batch_size=100,
                 return_reconstruction=False,
                 W_regularizer=l2(.01),
                 activity_regularizer=None):

        super(SparseCoding, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_reconstruction = return_reconstruction
        self.input = T.matrix()

        self.W = self.init((self.output_dim, self.input_dim))
        self.params = [self.W, ]

        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)
        if activity_regularizer:
            activity_regularizer.set_layer(self)
            self.regularizers.append(activity_regularizer)

    def _step(self, x_t, accum_1, accum_2, inputs):
        outputs = self.activation(T.dot(x_t, self.W))
        rec_error = T.sqr(inputs - outputs).sum()
        l1_norm = (self.gamma * diff_abs(x_t)).sum()
        cost = rec_error + l1_norm
        x, new_accum_1, new_accum_2 = _RMSPropStep(cost, x_t, accum_1, accum_2)
        return x, new_accum_1, new_accum_2, outputs

    def get_output(self, train=False):
        inputs = self.get_input(train)
        initial_states = alloc_zeros_matrix(self.batch_size, self.output_dim)
        outputs, updates = theano.scan(
            self._step,
            sequences=[],
            outputs_info=[initial_states, ]*3 + [None, ],
            non_sequences=inputs,
            n_steps=self.n_steps,
            truncate_gradient=self.truncate_gradient)

        if self.return_reconstruction:
            return outputs[-1][-1]
        else:
            return outputs[0][-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_reconstruction": self.return_reconstruction}


class ConvSparseCoding(Layer):
    def __init__(self, nb_filter, stack_size, nb_row, nb_col,
                 input_row, input_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 W_regularizer=l2(.01), activity_regularizer=None):

        super(ConvSparseCoding, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.nb_filter = nb_filter
        self.stack_size = stack_size

        self.nb_row = nb_row
        self.nb_col = nb_col
        if border_mode == 'valid':
            self.code_row = input_row + nb_row - 1
            self.code_col = input_col + nb_col - 1

        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W.shape)

        self.params = [self.W]

        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, accum_1, accum_2, inputs):
        conv_out = T.nnet.conv.conv2d(x_t, self.W, border_mode=self.border_mode,
                                      subsample=self.subsample)
        outputs = self.activation(conv_out)
        rec_error = T.sqr(inputs - outputs).sum()
        l1_norm = (self.gamma * diff_abs(x_t)).sum()
        cost = rec_error + l1_norm
        x, new_accum_1, new_accum_2 = _RMSPropStep(cost, x_t, accum_1, accum_2)
        return x, new_accum_1, new_accum_2, outputs

    def get_output(self, train):
        inputs = self.get_input(train)
        initial_states = alloc_zeros_matrix(self.batch_size, self.stack_size,
                                            self.code_row, self.code_col)
        outputs, updates = theano.scan(
            self._step,
            sequences=[],
            outputs_info=[initial_states, ]*3 + [None, ],
            non_sequences=inputs,
            n_steps=self.n_steps,
            truncate_gradient=self.truncate_gradient)

        if self.return_reconstruction:
            return outputs[-1][-1]
        else:
            return outputs[0][-1]
