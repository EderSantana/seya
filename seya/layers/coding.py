import theano
import theano.tensor as T
import numpy as np
import math
import scipy.sparse.linalg

from keras.layers.core import Layer
from keras.layers.recurrent import Recurrent
from keras import activations, initializations
from keras.utils.theano_utils import alloc_zeros_matrix, shared_scalar
from keras.regularizers import l2

from ..utils import diff_abs

floatX = theano.config.floatX


def _RMSPropStep(cost, states, accum_1, accum_2):
    rho = .9
    lr = .009
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
        initial_states = self.get_initial_states(inputs)
        outputs, updates = theano.scan(
            self._step,
            sequences=[],
            outputs_info=[initial_states, ]*3 + [None, ],
            non_sequences=[inputs, prior],
            n_steps=self.n_steps,
            truncate_gradient=self.truncate_gradient)

        outs = outputs[0][-1]
        if self.return_reconstruction:
            return T.dot(outs, self.W)
        else:
            return outs

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
        # causes_norm = diff_abs(u_tm1).sum() * self.gamma / 10.
        cost = rec_error + l1_norm + l1_inov  # + causes_norm
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


class SparseCodingFista(Layer):
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform',
                 activation='linear',
                 truncate_gradient=-1,
                 gamma=.1,
                 n_steps=100,
                 batch_size=100,
                 return_reconstruction=False,
                 W_regularizer=l2(.01),
                 activity_regularizer=None):

        super(SparseCodingFista, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.n_steps = n_steps
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_reconstruction = return_reconstruction
        self.batch_size = batch_size
        self.input = T.matrix()

        self.W = self.init((self.output_dim, self.input_dim))
        self.X = self.init((self.batch_size, output_dim))
        self.params = [self.W, ]

        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)
        if activity_regularizer:
            activity_regularizer.set_layer(self)
            self.regularizers.append(activity_regularizer)

        inputs = self.init((self.batch_size, self.input_dim))
        cost = T.sqr(inputs - T.dot(self.X, self.W)).sum()
        self._fista = Fista(cost, self.X, self.W, inputs, self)

    # def _fista(self, X):
    #     Phi = self.W.get_value().T
    #     Xnew = fista(X, Phi, max_iterations=self.n_steps).astype(floatX)
    #     self.X.set_value(Xnew.T)

    def get_output(self, train=False):
        inputs = self.get_input(train)
        if self.return_reconstruction:
            return T.dot(self.X, self.W) + inputs.sum()*0
        else:
            return self.X + inputs.sum()*0

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "return_reconstruction": self.return_reconstruction}


class Fista(object):
    def __init__(self, cost, X, params, inputs, model, lambdav=.1,
                 max_iter=100):
        """Fista optimization using Theano

        cost: TODO
        params: TODO
        lambdav: TODO
        max_iter: TODO

        """
        self.inputs = inputs
        self.cost = cost
        self.params = params
        self.lambdav = lambdav
        self.max_iter = max_iter
        self.X = X
        self.grads = T.grad(cost, params)
        self.updates = []

        Phi = params.get_value().T
        Q = Phi.T.dot(Phi)
        L = scipy.sparse.linalg.eigsh(2*Q, 1, which='LM')[0]
        invL = 1/float(L)

        self.y = model.init((model.batch_size, model.input_dim))
        self.t = model.init((1,))

        x2 = self._proxOp(self.y-invL*self.grads, invL*self.lambdav)
        t2 = .5 + T.sqrt(1+4*(self.t**2))/2.
        self.updates.append((self.y, x2 + ((self.t-1)/t2)*(x2-self.X)))
        self.updates.append((self.X, x2))
        self.updates.append((self.t, t2))

        self.F = theano.function([], [], updates=self.updates,
                                 allow_input_downcast=True)

    def optimize(self, x_batch):
        self.inputs.set_value(x_batch.astype(floatX))
        for i in range(self.max_iter):
            self.F()

    def _proxOp(self, x, t):
        return T.maximum(x-t, 0) + T.minimum(x+t, 0)


def fista(cost, params, Phi, lambdav=.1, max_iterations=150, display=False):
    """ FISTA Inference for Lasso (l1) Problem
    I: Batches of images (dim x batch)
    Phi: Dictionary (dim x dictionary element) (nparray or sparse array)
    lambdav: Sparsity penalty
    max_iterations: Maximum number of iterations
    """
    def proxOp(x, t):
        """ L1 Proximal Operator """
        return np.fmax(x-t, 0) + np.fmin(x+t, 0)

    x = np.zeros((Phi.shape[1], I.shape[1]))
    Q = Phi.T.dot(Phi)
    c = -2*Phi.T.dot(I)

    L = scipy.sparse.linalg.eigsh(2*Q, 1, which='LM')[0]
    invL = 1/float(L)

    y = x
    t = 1

    for i in range(max_iterations):
        g = 2*Q.dot(y) + c
        x2 = proxOp(y-invL*g, invL*lambdav)
        t2 = (1+math.sqrt(1+4*(t**2)))/2.0
        y = x2 + ((t-1)/t2)*(x2-x)
        x = x2
        t = t2

    return x2
