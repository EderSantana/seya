import theano.tensor as T

from theano import scan

from keras.layers.recurrent import Recurrent
from keras import activations, initializations, regularizers, constraints
from keras.utils.theano_utils import shared_zeros

from ..utils import apply_layer


class FDPCN(Recurrent):
    '''Fast DPCN layer
    '''
    def __init__(self, input_dim, states_dim, causes_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', gate_activation='sigmoid',
                 weights=None, return_mode='states',
                 truncate_gradient=-1, return_sequences=False):
        super(FDPCN, self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.states_dim = states_dim
        self.causes_dim = causes_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.gate_activation = activations.get(gate_activation)
        self.return_sequences = return_sequences
        self.return_mode = return_mode
        self.input = T.tensor3()

        self.I2S = self.init((self.input_dim, self.states_dim))
        self.S2S = self.inner_init((self.states_dim, self.states_dim))
        self.Sb = shared_zeros((self.states_dim))

        self.S2C = self.init((self.states_dim, self.causes_dim))
        self.C2C = self.inner_init((self.causes_dim, self.causes_dim))
        self.Cb = shared_zeros((self.causes_dim))
        self.CbS = shared_zeros((self.states_dim))
        self.C2S = self.init((self.causes_dim, self.states_dim))
        self.trainable_weights = [self.I2S, self.S2S, self.Sb,
                       self.C2S, self.C2C, self.Cb, self.S2C, self.CbS]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, mask_tm1, s_tm1, c_tm1,
              S2S, S2C, C2C, Cb, C2S, CbS):
        s_t = self.activation(x_t + mask_tm1 * T.dot(s_tm1, S2S))
        g = self.gate_activation(T.dot(c_tm1, C2S) + CbS)
        s_t *= g
        c_t = self.activation(T.dot(s_t, S2C) + T.dot(c_tm1, C2C) + Cb)
        return s_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle(1, 0, 2)
        x = T.dot(X, self.I2S) + self.Sb

        s_init = T.zeros((X.shape[1], self.states_dim))
        u_init = T.ones((X.shape[1], self.causes_dim)) * .001

        outputs, uptdates = scan(
            self._step,
            sequences=[x, dict(input=padded_mask, taps=[-1])],
            outputs_info=[s_init, u_init],
            non_sequences=[self.S2S, self.S2C, self.C2C, self.Cb, self.C2S,
                           self.CbS],
            truncate_gradient=self.truncate_gradient)

        if self.return_mode == 'both':
            return T.concatenate([outputs[0], outputs[1]],
                                 axis=-1)
        elif self.return_mode == 'states':
            out = outputs[0]
        elif self.return_mode == 'causes':
            out = outputs[1]
        else:
            raise ValueError("return_model {0} not valid. Choose "
                             "'both', 'states' or 'causes'".format(
                                 self.return_mode))
        if self.return_sequences:
            return out.dimshuffle(1, 0, 2)
        else:
            return out[-1]


class Tensor(Recurrent):
    '''Tensor class
    Motivated by the Fast Approximate DPCN model

    Parameters:
    ===========
    *_dim defines the dimensions of the tensorial transition
    hid2output: is a sequential model to transform the hidden states
        to the output causes.
    '''
    def __init__(self, input_dim, output_dim, causes_dim,
                 hid2output,
                 init='glorot_uniform',
                 W_regularizer=None,
                 W_constraint=None,
                 b_regularizer=None,
                 b_constraint=None,
                 activation=lambda X: T.minimum(20, T.maximum(0, X)),
                 activity_regularizer=None,
                 truncate_gradient=-1,
                 weights=None, name=None,
                 return_mode='both',
                 return_sequences=True):
        super(Tensor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.causes_dim = causes_dim
        self.activation = activations.get(activation)
        self.hid2output = hid2output
        self.init = initializations.get(init)
        self.truncate_gradient = truncate_gradient
        self.input = T.tensor3()
        self.return_mode = return_mode
        self.return_sequences = return_sequences

        self.W = self.init((input_dim, causes_dim, output_dim))
        self.C = self.init((output_dim, output_dim))
        self.b = shared_zeros((self.output_dim))

        self.trainable_weights = [self.W, self.C, self.b] + hid2output.trainable_weights

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W.name = '%s_W' % name
        self.C.name = '%s_C' % name

    def _step(self, Wx_t, s_tm1, u_tm1, C, b, *args):
        uWx = (u_tm1[:, :, None] * Wx_t).sum(axis=1)  # shape: batch/output_dim
        s_t = self.activation(uWx + T.dot(s_tm1, C) + b)
        u_t = apply_layer(self.hid2output, s_t)
        return s_t, u_t

    def get_output(self, train=False):
        X = self.get_input()
        Wx = T.tensordot(X, self.W, axes=(2, 0)).dimshuffle(1, 0, 2, 3)
        s_init = T.zeros((X.shape[0], self.output_dim))
        u_init = T.ones((X.shape[0], self.causes_dim)) / self.causes_dim
        outputs, uptdates = scan(
            self._step,
            sequences=[Wx],
            outputs_info=[s_init, u_init],
            non_sequences=[self.C, self.b] + self.hid2output.trainable_weights,
            truncate_gradient=self.truncate_gradient)

        if self.return_mode == 'both':
            return T.concatenate([outputs[0], outputs[1]],
                                 axis=-1)
        elif self.return_mode == 'states':
            out = outputs[0]
        elif self.return_mode == 'causes':
            out = outputs[1]
        else:
            raise ValueError("return_model {0} not valid. Choose "
                             "'both', 'states' or 'causes'".format(
                                 self.return_mode))

        if self.return_sequences:
            return out.dimshuffle(1, 0, 2)
        else:
            return out[-1]


class Tensor2(Tensor):
    '''Tensor class
    Motivated by the Fast Approximate DPCN model

    Parameters:
    ===========
    *_dim defines the dimensions of the tensorial transition
    hid2output: is a sequential model to transform the hidden states
        to the output causes.
    '''
    def __init__(self, *args, **kwargs):
        super(Tensor2, self).__init__(*args, **kwargs)
        del self.trainable_weights[1]

    def _step(self, Wx_t, s_tm1, u_tm1, b, *args):
        uWx = (u_tm1[:, :, None] * Wx_t).sum(axis=1)  # shape: batch/output_dim
        s_t = self.activation(uWx + b)
        u_t = apply_layer(self.hid2output, s_t)
        return s_t, u_t

    def get_output(self, train=False):
        X = self.get_input()
        Wx = T.tensordot(X, self.W, axes=(2, 0)).dimshuffle(1, 0, 2, 3)
        s_init = T.zeros((X.shape[0], self.output_dim))
        u_init = T.ones((X.shape[0], self.causes_dim)) / self.causes_dim
        outputs, uptdates = scan(
            self._step,
            sequences=[Wx],
            outputs_info=[s_init, u_init],
            non_sequences=[self.b] + self.hid2output.trainable_weights,
            truncate_gradient=self.truncate_gradient)

        if self.return_mode == 'both':
            return T.concatenate([outputs[0], outputs[1]],
                                 axis=-1)
        elif self.return_mode == 'states':
            out = outputs[0]
        elif self.return_mode == 'causes':
            out = outputs[1]
        else:
            raise ValueError("return_model {0} not valid. Choose "
                             "'both', 'states' or 'causes'".format(
                                 self.return_mode))

        if self.return_sequences:
            return out.dimshuffle(1, 0, 2)
        else:
            return out[-1]


class ProdTensor(Tensor):
    '''Tensor class
    Motivated by the Fast Approximate DPCN model

    Parameters:
    ===========
    *_dim defines the dimensions of the tensorial transition
    hid2output: is a sequential model to transform the hidden states
        to the output causes.
    '''
    def __init__(self, *args, **kwargs):
        super(ProdTensor, self).__init__(*args, **kwargs)
        self.W = self.init((self.input_dim, self.output_dim))
        self.C = self.init((self.causes_dim, self.output_dim))
        self.b0 = shared_zeros((self.output_dim))
        self.trainable_weights[0] = self.W
        self.trainable_weights[1] = self.C
        self.trainable_weights = self.trainable_weights + [self.b0, ]

    def _step(self, Wx_t, s_tm1, u_tm1, C, b0, b1, *args):
        Cu = self.activation(T.dot(u_tm1, C) + b0)
        s_t = self.activation(Wx_t*Cu + b1)
        u_t = apply_layer(self.hid2output, s_t)
        return s_t, u_t

    def get_output(self, train=False):
        X = self.get_input()
        Wx = T.dot(X, self.W).dimshuffle(1, 0, 2)
        s_init = T.zeros((X.shape[0], self.output_dim))
        u_init = T.ones((X.shape[0], self.causes_dim)) * .001
        outputs, uptdates = scan(
            self._step,
            sequences=[Wx],
            outputs_info=[s_init, u_init],
            non_sequences=[self.C, self.b0, self.b] + self.hid2output.trainable_weights,
            truncate_gradient=self.truncate_gradient)

        if self.return_mode == 'both':
            return T.concatenate([outputs[0], outputs[1]],
                                 axis=-1)
        elif self.return_mode == 'states':
            out = outputs[0]
        elif self.return_mode == 'causes':
            out = outputs[1]
        else:
            raise ValueError("return_model {0} not valid. Choose "
                             "'both', 'states' or 'causes'".format(
                                 self.return_mode))

        if self.return_sequences:
            return out.dimshuffle(1, 0, 2)
        else:
            return out[-1]


class ProdExp(Tensor):
    '''ProdExp class
    Motivated by the Fast Approximate DPCN model

    Parameters:
    ===========
    *_dim defines the dimensions of the tensorial transition
    hid2output: is a sequential model to transform the hidden states
        to the output causes.
    '''
    def __init__(self, *args, **kwargs):
        super(ProdExp, self).__init__(*args, **kwargs)
        del self.trainable_weights[1]

    def _step(self, Wx_t, s_tm1, u_tm1, b, *args):
        uWx = (u_tm1[:, :, None] * Wx_t).prod(axis=1)  # shape: batch/output_dim
        s_t = self.activation(uWx + b)
        u_t = apply_layer(self.hid2output, s_t)
        return s_t, u_t

    def get_output(self, train=False):
        X = self.get_input()
        Wx = T.tensordot(X, self.W, axes=(2, 0)).dimshuffle(1, 0, 2, 3)
        s_init = T.zeros((X.shape[0], self.output_dim))
        u_init = T.ones((X.shape[0], self.causes_dim)) / self.causes_dim
        outputs, uptdates = scan(
            self._step,
            sequences=[Wx],
            outputs_info=[s_init, u_init],
            non_sequences=[self.b] + self.hid2output.trainable_weights,
            truncate_gradient=self.truncate_gradient)

        if self.return_mode == 'both':
            return T.concatenate([outputs[0], outputs[1]],
                                 axis=-1)
        elif self.return_mode == 'states':
            out = outputs[0]
        elif self.return_mode == 'causes':
            out = outputs[1]
        else:
            raise ValueError("return_model {0} not valid. Choose "
                             "'both', 'states' or 'causes'".format(
                                 self.return_mode))

        if self.return_sequences:
            return out.dimshuffle(1, 0, 2)
        else:
            return out[-1]


class GAE(Recurrent):
    '''GAE class
    Motivated by the Fast Approximate DPCN model

    Parameters:
    ===========
    *_dim defines the dimensions of the tensorial transition
    hid2output: is a sequential model to transform the hidden states
        to the output causes.
    '''
    def __init__(self, input_dim, output_dim, causes_dim,
                 hid2output,
                 init='glorot_uniform',
                 W_regularizer=None,
                 W_constraint=None,
                 b_regularizer=None,
                 b_constraint=None,
                 activation=lambda X: T.minimum(20, T.maximum(0, X)),
                 activity_regularizer=None,
                 truncate_gradient=-1,
                 weights=None, name=None,
                 return_mode='both',
                 return_sequences=True):
        super(GAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.causes_dim = causes_dim
        self.activation = activations.get(activation)
        self.init = initializations.get(init)
        self.truncate_gradient = truncate_gradient
        self.input = T.tensor3()
        self.return_mode = return_mode
        self.return_sequences = return_sequences

        self.V = self.init((input_dim, output_dim))
        self.U = self.init((input_dim, output_dim))
        self.W = self.init((output_dim, causes_dim))
        self.bo = shared_zeros((self.output_dim))
        self.bc = shared_zeros((self.causes_dim))

        self.trainable_weights = [self.V, self.U, self.W]

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W.name = '%s_W' % name
        self.C.name = '%s_C' % name

    def _step(self, x_t, Vx_t, x_tm1, s_tm1, m_tm1,
              V, U, W):
        m = self.activation(T.dot(
            T.dot(x_tm1, U) * Vx_t, W))
        s_t = T.dot(T.dot(x_t, U) * T.dot(m, W.T), V.T)
        return x_t, s_t, m

    def get_output(self, train=False):
        X = self.get_input().dimshuffle(1, 0, 2)
        Vx = T.dot(X, self.V)
        x_init = T.zeros((X.shape[1], self.input_dim))
        s_init = T.zeros((X.shape[1], self.output_dim))
        u_init = T.zeros((X.shape[1], self.causes_dim))
        outputs, uptdates = scan(
            self._step,
            sequences=[X, Vx],
            outputs_info=[x_init, s_init, u_init],
            non_sequences=self.trainable_weights,
            truncate_gradient=self.truncate_gradient)

        if self.return_mode == 'both':
            return T.concatenate([outputs[1], outputs[2]],
                                 axis=-1)
        elif self.return_mode == 'states':
            out = outputs[1]
        elif self.return_mode == 'causes':
            out = outputs[2]
        else:
            raise ValueError("return_model {0} not valid. Choose "
                             "'both', 'states' or 'causes'".format(
                                 self.return_mode))

        if self.return_sequences:
            return out.dimshuffle(1, 0, 2)
        else:
            return out[-1]
