import theano
import theano.tensor as T

from keras.layers.core import Layer
from keras.layers.recurrent import Recurrent
from keras import activations, initializations
from keras.utils.theano_utils import alloc_zeros_matrix
from keras.regularizers import l2

from ..utils import diff_abs

floatX = theano.config.floatX


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
                 return_reconstruction=False,
                 W_regularizer=l2(.01),
                 activity_regularizer=None):

        super(SparseCoding, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.n_steps = n_steps
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

    def get_initial_states(self, X):
        return alloc_zeros_matrix(X.shape[0], self.output_dim)

    def _step(self, x_t, accum_1, accum_2, inputs, prior):
        outputs = self.activation(T.dot(x_t, self.W))
        rec_error = T.sqr(inputs - outputs).sum()
        l1_norm = (self.gamma * diff_abs(x_t)).sum()
        l1_inov = diff_abs(x_t - prior).sum() * self.gamma / 10.
        cost = rec_error + l1_norm + l1_inov
        x, new_accum_1, new_accum_2 = _RMSPropStep(cost, x_t, accum_1, accum_2)
        return x, new_accum_1, new_accum_2, outputs

    def _get_output(self, inputs, train=False, prior=0.):
        initial_states = self.get_initial_states()
        outputs, updates = theano.scan(
            self._step,
            sequences=[],
            outputs_info=[initial_states, ]*3 + [None, ],
            non_sequences=[inputs, prior],
            n_steps=self.n_steps,
            truncate_gradient=self.truncate_gradient)

        if self.return_reconstruction:
            return outputs[-1][-1]
        else:
            return outputs[0][-1]

    def get_output(self, train=False):
        inputs = self.get_input(train)
        return self._get_output(inputs, train)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_reconstruction": self.return_reconstruction}


class VarianceCoding(Layer):
    def __init__(self, input_dim, output_dim,
                 init='uniform',
                 truncate_gradient=-1,
                 gamma=.1,
                 n_steps=10,
                 W_regularizer=l2(.01),
                 activity_regularizer=None):

        super(VarianceCoding, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.n_steps = n_steps
        self.truncate_gradient = truncate_gradient
        self.activation = lambda x: .5*(1 + T.exp(-x))
        self.input = T.matrix()

        self.W = self.init((self.output_dim, self.input_dim))
        self.params = [self.W]

        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)
        if activity_regularizer:
            activity_regularizer.set_layer(self)
            self.regularizers.append(activity_regularizer)

    def get_initial_states(self, X):
        return alloc_zeros_matrix(X.shape[0], self.output_dim)

    def _step(self, x_t, accum_1, accum_2, inputs, prior):
        outputs = self.activation(T.dot(x_t, self.W))
        # rec_error = T.sqr(inputs - outputs).sum()
        l1_norm = self.gamma * outputs * diff_abs(inputs)
        own_norm = (diff_abs(x_t)).sum() * self.gamma
        l1_inov = T.sqr(x_t - prior).sum() * self.gamma / 10.
        cost = l1_norm.sum() + l1_inov + own_norm
        x, new_accum_1, new_accum_2 = _RMSPropStep(cost, x_t, accum_1, accum_2)
        return x, new_accum_1, new_accum_2, outputs, l1_norm

    def _get_output(self, inputs, train=False, prior=0.):
        initial_states = self.get_initial_states(inputs)
        outputs, updates = theano.scan(
            self._step,
            sequences=[],
            outputs_info=[initial_states, ]*3 + [None, None],
            non_sequences=[inputs, prior],
            n_steps=self.n_steps,
            truncate_gradient=self.truncate_gradient)
        return outputs[-1][-1]

    def get_output(self, train=False):
        inputs = self.get_input(train)
        return self._get_output(inputs, train)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_reconstruction": self.return_reconstruction}


class Sparse2L(Layer):
    '''A combined Sparse Coding + Variance Compoenent Layer
    '''
    def __init__(self, input_dim, output_dim, causes_dim,
                 init='glorot_uniform',
                 activation='linear',
                 truncate_gradient=-1,
                 gamma=.1,
                 n_steps=10,
                 return_mode='states',
                 W_regularizer=l2(.01),
                 V_regularizer=l2(.01),
                 activity_regularizer=None):

        super(Sparse2L, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.causes_dim = causes_dim
        self.gamma = gamma
        self.n_steps = n_steps
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_mode = return_mode
        self.input = T.matrix()

        self.W = self.init((self.output_dim, self.input_dim))
        self.V = self.init((self.causes_dim, self.output_dim))
        self.params = [self.W, self.V]

        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)
            V_regularizer.set_param(self.V)
            self.regularizers.append(V_regularizer)
        if activity_regularizer:
            activity_regularizer.set_layer(self)
            self.regularizers.append(activity_regularizer)

    def get_initial_states(self, X):
        return (alloc_zeros_matrix(X.shape[0], self.output_dim),
                alloc_zeros_matrix(X.shape[0], self.causes_dim))

    def _step(self, x_tm1, accum_1, accum_2,
              u_tm1, accum_1_u, accum_2_u,
              inputs, prior, *args):
        outputs = self.activation(T.dot(x_tm1, self.W))
        rec_error = T.sqr(inputs - outputs).sum()
        causes = (1 + T.exp(-T.dot(u_tm1, self.V))) * .5
        l1_norm = (self.gamma * causes * diff_abs(x_tm1)).sum()
        l1_inov = diff_abs(x_tm1 - prior).sum() * self.gamma / 10.
        causes_norm = diff_abs(u_tm1).sum() * self.gamma / 10.
        cost = rec_error + l1_norm + l1_inov + causes_norm
        x, new_accum_1, new_accum_2 = _RMSPropStep(cost, x_tm1, accum_1,
                                                   accum_2)
        u, new_accum_1_u, new_accum_2_u = _RMSPropStep(cost, u_tm1, accum_1_u,
                                                       accum_2_u)
        return (x, new_accum_1, new_accum_2, u, new_accum_1_u, new_accum_2_u,
                outputs)

    def _get_output(self, inputs, train=False, prior=0.):
        x_init, u_init = self.get_initial_states(inputs)
        outputs, updates = theano.scan(
            self._step,
            sequences=[],
            outputs_info=[x_init]*3 + [u_init]*3 + [None, ],
            non_sequences=[inputs, prior] + self.params,
            n_steps=self.n_steps,
            truncate_gradient=self.truncate_gradient)

        if self.return_mode == 'rec':
            return outputs[-1][-1]
        elif self.return_mode == 'states':
            return outputs[0][-1]
        elif self.return_mode == 'causes':
            return outputs[3][-1]
        else:
            raise ValueError

    def get_output(self, train=False):
        inputs = self.get_input(train)
        return self._get_output(inputs, train)

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
                 W_regularizer=l2(.01), activity_regularizer=None,
                 return_reconstruction=False, n_steps=10, truncate_gradient=-1,
                 gamma=0.1):

        super(ConvSparseCoding, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.nb_filter = nb_filter
        self.stack_size = stack_size
        self.n_steps = n_steps
        self.truncate_gradient = truncate_gradient
        self.gamma = gamma

        self.nb_row = nb_row
        self.nb_col = nb_col
        if border_mode == 'valid':
            self.code_row = input_row + nb_row - 1
            self.code_col = input_col + nb_col - 1
        else:
            raise ValueError("boder_model {0} not implemented yet.".format(
                border_mode))

        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W_shape)

        self.params = [self.W]

        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)

        if weights is not None:
            self.set_weights(weights)

        self.return_reconstruction = return_reconstruction

    def get_initial_states(self, X):
        return alloc_zeros_matrix(X.shape[0], self.stack_size,
                                  self.code_row, self.code_col)

    def _step(self, x_t, accum_1, accum_2, inputs):
        conv_out = T.nnet.conv.conv2d(x_t, self.W, border_mode=self.border_mode,
                                      subsample=self.subsample)
        outputs = self.activation(conv_out)
        rec_error = T.sqr(inputs - outputs).sum()
        l1_norm = (self.gamma * diff_abs(x_t)).sum()
        cost = rec_error + l1_norm
        x, new_accum_1, new_accum_2 = _RMSPropStep(cost, x_t, accum_1, accum_2)
        return x, new_accum_1, new_accum_2, outputs

    def _get_output(self, inputs, train):
        initial_states = self.get_initial_states(inputs)
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

    def get_output(self, train):
        inputs = self.get_input(train)
        return self._get_output(inputs, train)


class ConvSparse2L(Layer):
    def __init__(self, nb_filter, stack_size, nb_row, nb_col,
                 input_row, input_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 W_regularizer=l2(.001), V_regularizer=l2(.001),
                 activity_regularizer=None,
                 return_reconstruction=False, n_steps=10, truncate_gradient=-1,
                 gamma=0.1):

        super(ConvSparseCoding, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.nb_filter = nb_filter
        self.stack_size = stack_size
        self.n_steps = n_steps
        self.truncate_gradient = truncate_gradient
        self.gamma = gamma

        self.nb_row = nb_row
        self.nb_col = nb_col
        if border_mode == 'valid':
            self.code_row = input_row + nb_row - 1
            self.code_col = input_col + nb_col - 1
        else:
            raise ValueError("boder_model {0} not implemented yet.".format(
                border_mode))

        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W_shape)

        self.params = [self.W]

        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)

        if weights is not None:
            self.set_weights(weights)

        self.return_reconstruction = return_reconstruction

    def get_initial_states(self, X):
        return alloc_zeros_matrix(X.shape[0], self.stack_size,
                                  self.code_row, self.code_col)

    def _step(self, x_t, accum_1, accum_2, inputs):
        conv_out = T.nnet.conv.conv2d(x_t, self.W, border_mode=self.border_mode,
                                      subsample=self.subsample)
        outputs = self.activation(conv_out)
        rec_error = T.sqr(inputs - outputs).sum()
        l1_norm = (self.gamma * diff_abs(x_t)).sum()
        cost = rec_error + l1_norm
        x, new_accum_1, new_accum_2 = _RMSPropStep(cost, x_t, accum_1, accum_2)
        return x, new_accum_1, new_accum_2, outputs

    def _get_output(self, inputs, train):
        initial_states = self.get_initial_states(inputs)
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

    def get_output(self, train):
        inputs = self.get_input(train)
        return self._get_output(inputs, train)


class TemporalSparseCoding(Recurrent):
    def __init__(self, prototype, transition_net, truncate_gradient=-1,
                 return_reconstruction=True,
                 init='glorot_uniform'):

        super(TemporalSparseCoding, self).__init__()
        self.init = initializations.get(init)
        self.prototype = prototype
        self.W = prototype.W  # Sparse coding parameter I - Wx
        self.regularizers = prototype.regularizers
        self.activation = prototype.activation
        self.tnet = transition_net
        try:
            self.is_conv = False
            self.input_dim = prototype.input_dim
            self.output_dim = prototype.output_dim
            self.A = self.init((
                self.output_dim, self.output_dim))  # Predictive transition x_t - Ax_t-1
            self.input = T.tensor3()
        except:
            self.is_conv = True
            self.nb_filter = prototype.nb_filter
            self.stack_size = prototype.stack_size
            self.nb_row = prototype.nb_row
            self.nb_col = prototype.nb_col
            self.A = self.init(self.W.get_value().shape)
            self.input = T.TensorType(floatX, (False,)*5)()

        self.params = prototype.params  # + [self.A, ]
        self.return_reconstruction = return_reconstruction
        self.truncate_gradient = truncate_gradient

    def _step(self, inputs, x_t):
        tmp = self.tnet.input
        self.tnet.input = x_t
        prior = self.tnet.get_output()
        self.tnet.input = tmp
        if self.is_conv:
            '''
            prior = T.nnet.conv.conv2d(x_t, self.A, border_mode='full',
                                       subsample=self.subsample)
            br = slice(np.ceil(self.nb_row/2. - 1), np.floor(self.nb_row/2. - 1))
            bc = slice(np.ceil(self.nb_col/2. - 1), np.floor(self.nb_col/2. - 1))
            prior = prior[:, :, br, bc]
            '''
            new_x = self.prototype._get_output(inputs, prior=prior)
            inp = T.nnet.conv.conv2d(new_x, self.W, border_mode=self.border_mode,
                                     subsample=self.subsample)
            outputs = self.activation(inp)
        else:
            # prior = T.dot(x_t, self.A)
            new_x = self.prototype._get_output(inputs, prior=prior)
            outputs = self.activation(T.dot(new_x, self.W))
        return new_x, outputs

    def get_output(self, train=False):
        inputs = self.get_input(train).dimshuffle(1, 0, 2)
        initial_states = self.prototype.get_initial_states(inputs[0])
        # initial_states = alloc_zeros_matrix(self.batch_size, self.output_dim)
        outputs, updates = theano.scan(
            self._step,
            sequences=[inputs],
            outputs_info=[initial_states, None],
            truncate_gradient=self.truncate_gradient)

        if self.return_reconstruction:
            return outputs[-1].dimshuffle(1, 0, 2)
        else:
            return outputs[0].dimshuffle(1, 0, 2)

    def get_config(self):
        return {"name": self.__class__.__name__,
                # "input_dim": self.input_dim,
                # "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_reconstruction": self.return_reconstruction}
