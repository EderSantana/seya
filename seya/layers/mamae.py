import theano.tensor as T
from theano import scan

from keras.layers.recurrent import GRU, Recurrent
from keras.utils.theano_utils import shared_zeros


class MAMAE(Recurrent):
    '''MAMAE: Memory AugMented AutoEncoders

    Parameters:
    ===========
    h_dim : encoder dimension
    mem_shape : (rows, cols)
    N_read : Size of the read patch (default: 2)
    inner_rnn : str with rnn type ('gru' default)
    truncate_gradient : int (-1 default)
    return_sequences : bool (False default)
    '''
    def __init__(self, input_dim, mem_shape, h_dim, N_read=2,
                 inner_rnn='gru', truncate_gradient=-1, return_sequences=False):
        self.input_dim = input_dim
        self.h_dim = h_dim  # this is 256 for MNIST
        self.mem_shape = mem_shape
        self.N_read = N_read
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.height = mem_shape[0]
        self.width = mem_shape[1]

        self.inner_rnn = inner_rnn
        if inner_rnn == 'gru':
            self.rnn = GRU(input_dim=(self.N_read**2 + 5 + input_dim),
                           output_dim=h_dim)
            # NOTE: this "5" is the number of attention parameters in the
            # previous step
        else:
            raise ValueError('This type of rnn is not supported')

        self.init_h = shared_zeros((h_dim))  # initial values
        self.init_ap = shared_zeros((5))  # initial values
        self.W_read = self.rnn.init((5 + h_dim, 5))  # "read" attention parameters
        self.b_read = shared_zeros((5))
        self.params = self.rnn.params + [self.W_read, self.b_read, self.init_h,
                                         self.init_ap]

    def _get_attention_params(self, h, L, b, N):
        p = T.tanh(T.dot(h, L) + b)
        gx = self.width * (p[:, 0]+1) / 2.
        gy = self.height * (p[:, 1]+1) / 2.
        sigma2 = T.exp(p[:, 2])
        delta = T.exp(p[:, 3] * (max(self.width, self.height) - 1) / (N - 1))
        gamma = T.exp(p[:, 4])
        return gx, gy, sigma2, delta, gamma, p

    def _get_filterbank(self, gx, gy, sigma2, delta, N):
        eps = 1e-6
        i = T.arange(N)
        a = T.arange(self.width)
        b = T.arange(self.height)

        mx = gx[:, None] + delta[:, None] * (i - N/2 - .5)
        my = gy[:, None] + delta[:, None] * (i - N/2 - .5)

        Fx = T.exp(-(a - mx[:, :, None])**2 / 2. / sigma2[:, None, None])
        Fx /= (Fx.sum(axis=-1)[:, :, None] + eps)
        Fy = T.exp(-(b - my[:, :, None])**2 / 2. / sigma2[:, None, None])
        Fy /= (Fy.sum(axis=-1)[:, :, None] + eps)
        return Fx, Fy

    def _read(self, x, gamma, Fx, Fy):
        Fyx = (Fy[:, :, :, None] * x[:, None, :, :]).sum(axis=2)
        FxT = Fx.dimshuffle(0, 2, 1)
        FyxFx = (Fyx[:, :, :, None] * FxT[:, None, :, :]).sum(axis=2)
        return gamma[:, None, None] * FyxFx

    def _get_rnn_input(self, x, rnn):
        if self.inner_rnn == 'gru':
            x_z = T.dot(x, rnn.W_z) + rnn.b_z
            x_r = T.dot(x, rnn.W_r) + rnn.b_r
            x_h = T.dot(x, rnn.W_h) + rnn.b_h
        return x_z, x_r, x_h

    def _get_rnn_state(self, rnn, *args):
        if self.inner_rnn == 'gru':
            x_z, x_r, x_h, h_tm1, mask = args
            h = rnn._step(x_z, x_r, x_h, mask, h_tm1,
                          rnn.U_z, rnn.U_r, rnn.U_h)
        return h

    def _get_initial_states(self, batch_size):
        init_h = self.init_h.dimshuffle('x', 0).repeat(batch_size, axis=0)
        init_ap = self.init_ap.dimshuffle('x', 0).repeat(batch_size, axis=0)
        fake_state = T.concatenate([init_h, init_ap], axis=-1)
        _, _, _, _, _, init_ap = self._get_attention_params(
            fake_state, self.W_read, self.b_read, self.N_read)
        return init_h, init_ap

    def _step(*args):
        raise NotImplemented('This is a base class')

    def get_output(*args, **kwargs):
        raise NotImplemented('This is a base class')


class MAMAEenc(MAMAE):
    '''MAMAE: Memory AugMented AutoEncoders (ENCODER)

    Parameters:
    ===========
    input_dim : dimension of the input time sequence
    h_dim : encoder dimension
    mem_shape : (rows, cols)
    N_read : Size of the read patch (default: 2)
    N_write : Size of the write pathc (default: 5)
    inner_rnn : str with rnn type ('gru' default)
    truncate_gradient : int (-1 default)
    return_sequences : bool (False default)
    '''

    def __init__(self, input_dim, mem_shape, h_dim, N_read=2, N_write=5,
                 inner_rnn='gru', truncate_gradient=-1, return_sequences=False,
                 canvas_activation=T.nnet.sigmoid):
        super(MAMAEenc, self).__init__(input_dim, mem_shape, h_dim, N_read=N_read,
                                       inner_rnn=inner_rnn,
                                       truncate_gradient=truncate_gradient,
                                       return_sequences=return_sequences)
        self.input = T.tensor3()
        self.N_write = N_write

        self.init_mem = shared_zeros(mem_shape)  # memory and hidden state
        self.W_write = self.rnn.init((h_dim, self.N_write**2))
        self.b_write = shared_zeros((self.N_write**2))
        self.params += [self.W_write, self.b_write, self.init_mem]

    def _write(self, h, gamma, Fx, Fy):
        write_patch = T.tanh(T.dot(h, self.W_write) + self.b_write)
        write_patch = write_patch.reshape((
            h.shape[0], self.N_write, self.N_write))
        Fyx = (Fy[:, :, :, None] * write_patch[:, :, None, :]).sum(axis=1)
        FyxFx = (Fyx[:, :, :, None] * Fx[:, None, :, :]).sum(axis=2)
        return FyxFx / gamma[:, None, None]

    def _get_initial_states(self, batch_size):
        init_h, init_ap = super(MAMAEenc, self)._get_initial_states(batch_size)
        mem = self.init_mem.dimshuffle('x', 0, 1).repeat(batch_size,
                                                         axis=0)
        return mem, init_h, init_ap

    def _step(self, x_t, mask, mem_tm1, h_tm1, ap_tm1, *args):
        read_state = T.concatenate([h_tm1, ap_tm1], axis=-1)
        gx, gy, sigma2, delta, gamma, ap_t = self._get_attention_params(
            read_state, self.W_read, self.b_read, self.N_read)
        Fx, Fy = self._get_filterbank(gx, gy, sigma2, delta, self.N_read)
        read_mem = self._read(mem_tm1, gamma, Fx, Fy).flatten(ndim=2)

        enc_input = T.concatenate([x_t, read_mem, ap_t], axis=-1)
        x_z, x_r, x_h = self._get_rnn_input(enc_input, self.rnn)

        h_t = self._get_rnn_state(self.rnn, x_z, x_r, x_h, h_tm1, mask)

        gx_w, gy_w, sigma2_w, delta_w, gamma_w, _ = self._get_attention_params(
            h_t, self.W_write, self.b_write, self.N_write)
        Fx_w, Fy_w = self._get_filterbank(gx_w, gy_w, sigma2_w, delta_w,
                                          self.N_write)
        mem_t = mem_tm1 + self._write(h_t, gamma_w, Fx_w, Fy_w)
        return mem_t, h_t, ap_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle(1, 0, 2)
        mem, init_h, init_ap = self._get_initial_states(X.shape[1])

        # if self.inner_rnn == 'gru':
        #     X_z = T.dot(X, self.rnn.W_z) + self.rnn.b_z
        #     X_r = T.dot(X, self.rnn.W_r) + self.rnn.b_r
        #     X_h = T.dot(X, self.rnn.W_h) + self.rnn.b_h

        outputs, updates = scan(self._step,
                                sequences=[X, padded_mask],
                                outputs_info=[mem, init_h, init_ap],
                                non_sequences=self.params,
                                truncate_gradient=self.truncate_gradient)
        # self.updates = updates
        if self.return_sequences:
            return outputs[0].dimshuffle(1, 0, 2, 3)
        else:
            return outputs[0][-1]


class MAMAEdec(MAMAE):
    '''MAMAE: Memory AugMented AutoEncoders (DECODER)

    Parameters:
    ===========
    h_dim : encoder dimension
    z_dim : random sample dimension (reparametrization trick output)
    mem_shape : (rows, cols)
    n_steps : how many time samples should the decoder generate
    N_read : Size of the read patch (default: 2)
    inner_rnn : str with rnn type ('gru' default)
    truncate_gradient : int (-1 default)
    return_sequences : bool (False default)
    '''

    def __init__(self, mem_shape, h_dim, n_steps, N_read=2,
                 inner_rnn='gru', truncate_gradient=-1, return_sequences=False):
        input_dim = 0
        super(MAMAEdec, self).__init__(input_dim, mem_shape, h_dim, N_read=N_read,
                                       inner_rnn=inner_rnn,
                                       truncate_gradient=truncate_gradient,
                                       return_sequences=return_sequences)
        self.input = T.matrix()
        self.n_steps = n_steps

    def _step(self, h_tm1, ap_tm1, mem, *args):
        """ap_tm1 = attention parameters from previous step, i.e. it know where it
        read in the previous step"""
        read_state = T.concatenate([h_tm1, ap_tm1], axis=-1)
        gx, gy, sigma2, delta, gamma, ap_t = self._get_attention_params(
            read_state, self.W_read, self.b_read, self.N_read)
        Fx, Fy = self._get_filterbank(gx, gy, sigma2, delta, self.N_read)
        read_mem = self._read(mem, gamma, Fx, Fy).flatten(ndim=2)
        dec_input = T.concatenate([read_mem, ap_t], axis=-1)

        x_z_t, x_r_t, x_h_t = self._get_rnn_input(dec_input, self.rnn)
        mask_t = 1.
        # NOTE: no mask inside decoder, you should handle it at the
        # output masking in the last layer of the model. This will pass the input
        # masking though
        h_t = self._get_rnn_state(self.rnn, x_z_t, x_r_t, x_h_t, h_tm1, mask_t)

        return h_t, ap_t

    def get_output(self, train=False):
        self._train = train
        Mem = self.get_input(train)
        init_h, init_ap = self._get_initial_states(Mem.shape[0])

        outputs, updates = scan(self._step,
                                sequences=[],
                                outputs_info=[init_h, init_ap],
                                non_sequences=[Mem, ] + self.params,
                                n_steps=self.n_steps,
                                truncate_gradient=self.truncate_gradient)
        # self.updates = updates
        if self.return_sequences:
            return outputs[0].dimshuffle(1, 0, 2)
        else:
            return outputs[0][-1]
