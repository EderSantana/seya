import theano
import theano.tensor as T

from keras.layers.recurrent import GRU
from keras.utils.theano_utils import shared_zeros

from seya.utils import apply_model


def _masking(h_t, h_tm1, mask):
    mask = mask[:, 0].dimshuffle(0, 'x')
    return mask * h_t + (1-mask) * h_tm1


class GRUwithReadout(GRU):
    """
    GRUwithReadout
    GRU with a last layer whose output is also fedback with
    the dynamic state.

    Extra Parameter:
    ----------------
    readout: `keras.model.sequential`
    state_dim: dimension of the inner GRU

    Notes:
    ------
    output_dim == readout.output_dim

    """
    def __init__(self, readout,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1,
                 return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):

        self.readout = readout
        self.state_dim = readout.layers[0].input_shape[1]  # state_dim
        input_dim = readout.layers[0].input_shape[1]
        super(GRUwithReadout, self).__init__(
            self.state_dim,
            init=init, inner_init=inner_init,
            activation=activation, inner_activation=inner_activation,
            weights=weights, truncate_gradient=truncate_gradient,
            return_sequences=return_sequences, input_dim=input_dim,
            **kwargs)
        self.output_dim = self.readout.output_shape[1]

    def build(self):
        self.readout.build()
        self.init_h = shared_zeros((self.state_dim,))
        # here is difference on the sizes
        input_dim = self.input_shape[2] + self.readout.output_shape[1]

        # copy-paste from keras.recurrent.GRU
        self.W_z = self.init((input_dim, self.state_dim))
        self.U_z = self.inner_init((self.state_dim, self.state_dim))
        self.b_z = shared_zeros((self.state_dim))

        self.W_r = self.init((input_dim, self.state_dim))
        self.U_r = self.inner_init((self.state_dim, self.state_dim))
        self.b_r = shared_zeros((self.state_dim))

        self.W_h = self.init((input_dim, self.state_dim))
        self.U_h = self.inner_init((self.state_dim, self.state_dim))
        self.b_h = shared_zeros((self.state_dim))

        self.trainable_weights = [
            self.init_h,
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _get_initial_states(self, batch_size):
        init_h = T.repeat(self.init_h.dimshuffle('x', 0), batch_size, axis=0)
        init_o = apply_model(self.readout, init_h)
        return init_h, init_o

    def _step(self,
              x_t, mask_tm1,
              h_tm1, o_tm1,
              u_z, u_r, u_h, *args):
        xo = T.concatenate([x_t, o_tm1], axis=-1)
        xz_t = T.dot(xo, self.W_z) + self.b_z
        xr_t = T.dot(xo, self.W_r) + self.b_r
        xh_t = T.dot(xo, self.W_h) + self.b_h
        z = self.inner_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        o_t = apply_model(self.readout, h_t)

        return _masking(h_t, h_tm1, mask_tm1), _masking(o_t, o_tm1, mask_tm1)

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        init_h, init_o = self._get_initial_states(X.shape[1])
        outputs, updates = theano.scan(
            self._step,
            sequences=[X, padded_mask],
            outputs_info=[init_h, init_o],
            non_sequences=[self.U_z, self.U_r, self.U_h] + self.readout.trainable_weights,
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs[1].dimshuffle((1, 0, 2))
        return outputs[1][-1]
