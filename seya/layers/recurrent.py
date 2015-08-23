import types
import theano
import theano.tensor as T

from keras.layers.recurrent import Recurrent, GRU
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix


def _get_reversed_input(self, train=False):
    if hasattr(self, 'previous'):
        X = self.previous.get_output(train=train)
    else:
        X = self.input
    return X[::-1]


class Bidirectional(Recurrent):
    def __init__(self, forward, backward):
        super(Bidirectional, self).__init__()
        self.forward = forward
        self.backward = backward
        self.params = forward.params + backward.params
        self.input = T.tensor3()
        self.forward.input = self.input
        self.backward.input = self.input

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


class GRUM(GRU):
    def __init__(self, input_dim, output_dim=128, mem=None,
                 mem_dim=128, init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 return_mode='states'):

        super(GRUM, self).__init__(input_dim, output_dim, init=init,
                                   inner_init=inner_init, activation=activation,
                                   inner_activation=inner_activation,
                                   truncate_gradient=truncate_gradient,
                                   return_sequences=return_sequences)
        if mem is None:
            self.mem = shared_zeros((1, mem_dim))
        else:
            self.mem = mem
        self.mem_dim = mem_dim
        self.return_mode = return_mode

        self.Hm_z = self.init((self.mem_dim, self.output_dim))
        self.Hm_r = self.init((self.mem_dim, self.output_dim))
        self.Hm_h = self.init((self.mem_dim, self.output_dim))

        self.Wm_z = self.init((self.input_dim, self.mem_dim))
        self.Um_z = self.inner_init((self.mem_dim, self.mem_dim))
        self.Vm_z = self.inner_init((self.output_dim, self.mem_dim))
        self.bm_z = shared_zeros((self.mem_dim))

        self.Wm_r = self.init((self.input_dim, self.mem_dim))
        self.Um_r = self.inner_init((self.mem_dim, self.mem_dim))
        self.Vm_r = self.inner_init((self.output_dim, self.mem_dim))
        self.bm_r = shared_zeros((self.mem_dim))

        self.Wm_h = self.init((self.input_dim, self.mem_dim))
        self.Um_h = self.inner_init((self.mem_dim, self.mem_dim))
        self.Vm_h = self.inner_init((self.mem_dim, self.mem_dim))
        self.bm_h = shared_zeros((self.mem_dim))

        self.params = self.params + [
            self.Hm_z, self.Hm_r, self.Hm_h,
            self.Wm_z, self.Um_z, self.bm_z,
            self.Wm_r, self.Um_r, self.bm_r,
            self.Wm_h, self.Um_h, self.bm_h,
        ]

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              xzm_t, xrm_t, xhm_t,
              h_tm1, m_tm1,
              u_z, u_r, u_h, hm_z, hm_r, hm_h,
              vm_z, vm_r, vm_h, m_z, m_r, m_h
              ):
        h_mask_tm1 = mask_tm1 * h_tm1
        # solid state
        zm = self.inner_activation(xzm_t + T.dot(h_mask_tm1, vm_z)
                                   + T.dot(m_tm1, m_z)).means(axis=0)
        rm = self.inner_activation(xrm_t + T.dot(h_mask_tm1, vm_r)
                                   + T.dot(m_tm1, m_r)).mean(axis=0)
        mm_t = self.activation(xhm_t + T.dot(rm * m_tm1, vm_h)
                               + T.dot(m_tm1, m_h)).mean(axis=0)
        m_t = zm * m_tm1 + (1 - zm) * mm_t
        # short temr
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z)
                                  + T.dot(m_t, hm_z)[0])
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r)
                                  + T.dot(m_t, hm_r)[0])
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h)
                               + T.dot(m_t, hm_h)[0])
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t, m_t

    def get_output(self, train=False, mem_updates=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        xm_z = T.dot(X, self.Wm_z) + self.bm_z
        xm_r = T.dot(X, self.Wm_r) + self.bm_r
        xm_h = T.dot(X, self.Wm_h) + self.bm_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask, xm_z.mean(axis=0),
                       xm_r, xm_h],
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1],
                                                           self.output_dim), 1),
                          self.mem],
            non_sequences=[self.U_z, self.U_r, self.U_h, self.Hm_z, self.Hm_r,
                           self.Hm_h, self.Vm_z, self.Vm_r, self.Vm_h,
                           self.Um_z, self.Um_r, self.Um_h],
            truncate_gradient=self.truncate_gradient)

        self.mem_updates = outputs[1][-1]

        if mem_updates:
            return outputs[1][-1:]

        if self.return_sequences:
            out = outputs[0].dimshuffle((1, 0, 2))
        else:
            out = outputs[0][-1]

        if self.return_mode == 'states':
            return out
        elif self.return_mode == 'both':
            return [out, outputs[1][-1]]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "mem_dim": self.mem_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class ExoGRUM(GRUM):
    def __init__(self, input_dim, output_dim=128, mem=None,
                 mem_dim=128, init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 return_mode='states'):

        super(ExoGRUM, self).__init__(
            input_dim=input_dim, output_dim=output_dim, mem=mem,
            mem_dim=mem_dim, init=init, inner_init=inner_init,
            activation=activation, inner_activation=inner_activation,
            weights=weights, truncate_gradient=truncate_gradient,
            return_sequences=return_sequences, return_mode=return_mode)

    def get_output(self, train=False, mem_updates=False):
        X, mem = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        xm_z = T.dot(X, self.Wm_z) + self.bm_z
        xm_r = T.dot(X, self.Wm_r) + self.bm_r
        xm_h = T.dot(X, self.Wm_h) + self.bm_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask, xm_z.mean(axis=0),
                       xm_r.mean(axis=0), xm_h.mean(axis=0)],
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1],
                                                           self.output_dim), 1),
                          mem],
            non_sequences=[self.U_z, self.U_r, self.U_h, self.Hm_z, self.Hm_r,
                           self.Hm_h, self.Vm_z, self.Vm_r, self.Vm_h,
                           self.Um_z, self.Um_r, self.Um_h],
            truncate_gradient=self.truncate_gradient)

        self.mem_updates = outputs[1][-1]

        if mem_updates:
            return outputs[1][-1:]

        if self.return_sequences:
            out = outputs[0].dimshuffle((1, 0, 2))
        else:
            out = outputs[0][-1]

        if self.return_mode == 'states':
            return out
        elif self.return_mode == 'both':
            return [out, outputs[1][-1]]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "mem_dim": self.mem_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}
