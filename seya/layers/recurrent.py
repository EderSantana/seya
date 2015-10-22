import types
import theano
import theano.tensor as T

from keras.layers.recurrent import Recurrent, GRU
from keras.utils.theano_utils import shared_zeros


def _get_reversed_input(self, train=False):
    if hasattr(self, 'previous'):
        X = self.previous.get_output(train=train)
    else:
        X = self.input
    return X[::-1]


class Bidirectional(Recurrent):
    def __init__(self, forward, backward, return_sequences=False,
                 truncate_gradient=-1, **kwargs):
        super(Bidirectional, self).__init__(**kwargs)
        self.forward = forward
        self.backward = backward
        self.params = forward.params + backward.params
        self.input = T.tensor3()
        self.forward.input = self.input
        self.backward.input = self.input
        self.return_sequences = return_sequences
        self.truncate_gradient = truncate_gradient

        rs = (self.return_sequences, forward.return_sequences,
              backward.return_sequences)
        if rs[1:] != rs[:-1]:
            raise ValueError("Make sure 'return_sequences' is equal for self," +
                             " forward and backward.")
        tg = (self.truncate_gradient, forward.truncate_gradient,
              backward.truncate_gradient)
        if tg[1:] != tg[:-1]:
            raise ValueError("Make sure 'truncate_gradient' is equal for self," +
                             " forward and backward.")

    def set_previous(self, layer):
        if not self.supports_masked_input() and layer.get_output_mask() is not None:
            raise Exception("Attached non-masking layer to layer with masked output")
        self.previous = layer
        self.forward.previous = layer
        self.backward.previous = layer
        self.backward.get_input = types.MethodType(_get_reversed_input, self.backward)

    def get_output(self, train=False):
        Xf = self.forward.get_output(train)
        Xb = self.backward.get_output(train)
        Xb = Xb[::-1]
        return T.concatenate([Xf, Xb], axis=-1)

    def get_config(self):
        new_dict = {}
        for k, v in self.forward.get_cofig.items():
            new_dict['forward_'+k] = v
        for k, v in self.backward.get_cofig.items():
            new_dict['backward_'+k] = v
        new_dict["name"] = self.__class__.__name__
        return new_dict


class StatefulGRU(GRU):
    def __init__(self, input_dim, batch_size, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(StatefulGRU, self).__init__(
            input_dim, output_dim, init=init, inner_init=inner_init,
            activation=activation, inner_activation=inner_activation,
            weights=weights, truncate_gradient=truncate_gradient,
            return_sequences=return_sequences)

        self.h = shared_zeros((batch_size, output_dim))  # Here is the state

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
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)

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
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}
