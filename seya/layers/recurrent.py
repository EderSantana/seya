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
                 truncate_gradient=-1):
        super(Bidirectional, self).__init__()
        self.input = T.tensor3()
        self.forward = forward
        self.backward = backward
        self.return_sequences = return_sequences
        self.truncate_gradient = truncate_gradient
        self.output_dim = self.forward.output_dim
        if self.forward.output_dim != self.backward.output_dim:
            raise ValueError("Make sure `forward` and `backward` have " +
                             "the same `ouput_dim.`")

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

    def build(self):
        # self.forward.input = self.input
        # self.backward.input = self.input
        self.forward.build()
        self.backward.build()
        self.params = self.forward.params + self.backward.params

    def set_previous(self, layer, connection_map={}):
        assert self.nb_input == layer.nb_output == 1, "Cannot connect layers: input count and output count should be 1."
        if hasattr(self, 'input_ndim'):
            assert self.input_ndim == len(layer.output_shape), "Incompatible shapes: layer expected input with ndim=" +\
                str(self.input_ndim) + " but previous layer has output_shape " + str(layer.output_shape)
        self.forward.set_previous(layer, connection_map)
        self.backward.set_previous(layer, connection_map)
        self.backward.get_input = types.MethodType(_get_reversed_input, self.backward)
        self.previous = layer
        self.build()

    @property
    def output_shape(self):
        input_shape = self.input_shape
        f_out = self.forward.output_dim
        b_out = self.backward.output_dim
        if self.return_sequences:
            return (input_shape[0], input_shape[1], f_out + b_out)
        else:
            return (input_shape[0], f_out + b_out)

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
    def __init__(self, batch_size, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):

        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(StatefulGRU, self).__init__(
            output_dim, init=init, inner_init=inner_init,
            activation=activation, inner_activation=inner_activation,
            weights=weights, truncate_gradient=truncate_gradient,
            return_sequences=return_sequences,
            input_dim=input_dim, input_length=input_length, **kwargs)

    def build(self):
        super(StatefulGRU, self).build()
        self.h = shared_zeros((self.batch_size, self.output_dim))  # Here is the state

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
