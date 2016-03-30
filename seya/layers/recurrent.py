import types
import theano
import theano.tensor as T

from keras.layers.recurrent import Recurrent, GRU
from keras import backend as K


def _get_reversed_input(self, train=False):
    if hasattr(self, 'previous'):
        X = self.previous.get_output(train=train)
    else:
        X = self.input
    return X[::-1]


class Bidirectional(Recurrent):
    def __init__(self, forward=None, backward=None, return_sequences=False,
                 forward_conf=None, backward_conf=None):
        assert forward is not None or forward_conf is not None, "Must provide a forward RNN or a forward configuration"
        assert backward is not None or backward_conf is not None, "Must provide a backward RNN or a backward configuration"
        super(Bidirectional, self).__init__()
        if forward is not None:
            self.forward = forward
        else:
            # Must import inside the function, because in order to support loading
            # we must import this module inside layer_utils... ugly
            from keras.utils.layer_utils import container_from_config
            self.forward = container_from_config(forward_conf)
        if backward is not None:
            self.backward = backward
        else:
            from keras.utils.layer_utils import container_from_config
            self.backward = container_from_config(backward_conf)
        self.return_sequences = return_sequences
        self.output_dim = self.forward.output_dim + self.backward.output_dim

        if not (self.return_sequences == self.forward.return_sequences == self.backward.return_sequences):
            raise ValueError("Make sure 'return_sequences' is equal for self,"
                             " forward and backward.")

    def build(self):
        self.input = T.tensor3()
        self.forward.input = self.input
        self.backward.input = self.input
        self.forward.build()
        self.backward.build()
        self.trainable_weights = self.forward.trainable_weights + self.backward.trainable_weights

    def set_previous(self, layer):
        assert self.nb_input == layer.nb_output == 1, "Cannot connect layers: input count and output count should be 1."
        if hasattr(self, 'input_ndim'):
            assert self.input_ndim == len(layer.output_shape), "Incompatible shapes: layer expected input with ndim=" +\
                str(self.input_ndim) + " but previous layer has output_shape " + str(layer.output_shape)
        self.forward.set_previous(layer)
        self.backward.set_previous(layer)
        self.backward.get_input = types.MethodType(_get_reversed_input, self.backward)
        self.previous = layer
        self.build()

    @property
    def output_shape(self):
        input_shape = self.input_shape
        output_dim = self.output_dim
        if self.return_sequences:
            return (input_shape[0], input_shape[1], output_dim)
        else:
            return (input_shape[0], output_dim)

    def get_output(self, train=False):
        Xf = self.forward.get_output(train)
        Xb = self.backward.get_output(train)
        Xb = Xb[::-1]
        return T.concatenate([Xf, Xb], axis=-1)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'forward_conf': self.forward.get_config(),
                'backward_conf': self.backward.get_config(),
                'return_sequences': self.return_sequences}


class StatefulGRU(GRU):
    def __init__(self, batch_size, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):

        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(StatefulGRU, self).__init__(
            output_dim, init=init, inner_init=inner_init,
            activation=activation, inner_activation=inner_activation,
            weights=weights,
            return_sequences=return_sequences,
            input_dim=input_dim, input_length=input_length, **kwargs)

    def build(self):
        super(StatefulGRU, self).build()
        self.h = K.zeros((self.batch_size, self.output_dim))  # Here is the state

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=self.h[:X.shape[1]],
            non_sequences=[self.U_z, self.U_r, self.U_h])

        self.updates = ((self.h, outputs[-1]), )  # initial state of next batch
                                                  # is the last state of this
                                                  # batch
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def init_updates(self):
        self.get_output(train=True)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "return_sequences": self.return_sequences}
