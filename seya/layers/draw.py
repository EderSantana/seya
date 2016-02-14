import theano.tensor as T
from theano import scan

from keras.layers.recurrent import GRU, Recurrent, LSTM
from keras.utils.theano_utils import shared_zeros  # , alloc_zeros_matrix

from ..utils import theano_rng
from ..regularizers import SimpleCost


class DRAW(Recurrent):
    '''DRAW

    Parameters:
    ===========
    output_dim : encoder/decoder dimension
    code_dim : random sample dimension (reparametrization trick output)
    input_shape : (n_channels, rows, cols)
    N_enc : Size of the encoder's filter bank (MNIST default: 2)
    N_dec : Size of the decoder's filter bank (MNIST default: 5)
    n_steps : number of sampling steps (or how long it takes to draw, default 64)
    inner_rnn : str with rnn type ('gru' default)
    truncate_gradient : int (-1 default)
    return_sequences : bool (False default)
    '''
    theano_rng = theano_rng()

    def __init__(self, output_dim, code_dim, N_enc=2, N_dec=5, n_steps=64,
                 inner_rnn='gru', truncate_gradient=-1, return_sequences=False,
                 canvas_activation=T.nnet.sigmoid, init='glorot_uniform',
                 inner_init='orthogonal', input_shape=None, **kwargs):
        self.output_dim = output_dim  # this is 256 for MNIST
        self.code_dim = code_dim  # this is 100 for MNIST
        self.N_enc = N_enc
        self.N_dec = N_dec
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.n_steps = n_steps
        self.canvas_activation = canvas_activation
        self.init = init
        self.inner_init = inner_init
        self.inner_rnn = inner_rnn

        self.height = input_shape[1]
        self.width = input_shape[2]

        self._input_shape = input_shape
        super(DRAW, self).__init__(**kwargs)

    def build(self):
        self.input = T.tensor4()

        if self.inner_rnn == 'gru':
            self.enc = GRU(
                input_length=self.n_steps,
                input_dim=self._input_shape[0]*2*self.N_enc**2 + self.output_dim,
                output_dim=self.output_dim, init=self.init,
                inner_init=self.inner_init)
            self.dec = GRU(
                input_length=self.n_steps,
                input_dim=self.code_dim, output_dim=self.output_dim,
                init=self.init,
                inner_init=self.inner_init)

        elif self.inner_rnn == 'lstm':
            self.enc = LSTM(
                input_length=self.n_steps,
                input_dim=self._input_shape[0]*2*self.N_enc**2 + self.output_dim,
                output_dim=self.output_dim, init=self.init, inner_init=self.inner_init)
            self.dec = LSTM(
                input_length=self.n_steps,
                input_dim=self.code_dim, output_dim=self.output_dim,
                init=self.init, inner_init=self.inner_init)
        else:
            raise ValueError('This type of inner_rnn is not supported')

        self.enc.build()
        self.dec.build()

        self.init_canvas = shared_zeros(self._input_shape)  # canvas and hidden state
        self.init_h_enc = shared_zeros((self.output_dim))  # initial values
        self.init_h_dec = shared_zeros((self.output_dim))  # should be trained
        self.L_enc = self.enc.init((self.output_dim, 5))  # "read" attention parameters (eq. 21)
        self.L_dec = self.enc.init((self.output_dim, 5))  # "write" attention parameters (eq. 28)
        self.b_enc = shared_zeros((5))  # "read" attention parameters (eq. 21)
        self.b_dec = shared_zeros((5))  # "write" attention parameters (eq. 28)
        self.W_patch = self.enc.init((self.output_dim, self.N_dec**2*self._input_shape[0]))
        self.b_patch = shared_zeros((self.N_dec**2*self._input_shape[0]))
        self.W_mean = self.enc.init((self.output_dim, self.code_dim))
        self.W_sigma = self.enc.init((self.output_dim, self.code_dim))
        self.b_mean = shared_zeros((self.code_dim))
        self.b_sigma = shared_zeros((self.code_dim))
        self.trainable_weights = self.enc.trainable_weights + self.dec.trainable_weights + [
            self.L_enc, self.L_dec, self.b_enc, self.b_dec, self.W_patch,
            self.b_patch, self.W_mean, self.W_sigma, self.b_mean, self.b_sigma,
            self.init_canvas, self.init_h_enc, self.init_h_dec]

        if self.inner_rnn == 'lstm':
            self.init_cell_enc = shared_zeros((self.output_dim))     # initial values
            self.init_cell_dec = shared_zeros((self.output_dim))     # should be trained
            self.trainable_weights = self.trainable_weights + [self.init_cell_dec, self.init_cell_enc]

    def set_previous(self, layer, connection_map={}):
        self.previous = layer
        self.build()
        self.init_updates()

    def init_updates(self):
        self.get_output(train=True)  # populate regularizers list

    def _get_attention.trainable_weights(self, h, L, b, N):
        p = T.dot(h, L) + b
        gx = self.width * (p[:, 0]+1) / 2.
        gy = self.height * (p[:, 1]+1) / 2.
        sigma2 = T.exp(p[:, 2])
        delta = T.exp(p[:, 3]) * (max(self.width, self.height) - 1) / (N - 1.)
        gamma = T.exp(p[:, 4])
        return gx, gy, sigma2, delta, gamma

    def _get_filterbank(self, gx, gy, sigma2, delta, N):
        small = 1e-4
        i = T.arange(N)
        a = T.arange(self.width)
        b = T.arange(self.height)

        mx = gx[:, None] + delta[:, None] * (i - N/2. - .5)
        my = gy[:, None] + delta[:, None] * (i - N/2. - .5)

        Fx = T.exp(-(a - mx[:, :, None])**2 / 2. / sigma2[:, None, None])
        Fx /= (Fx.sum(axis=-1)[:, :, None] + small)
        Fy = T.exp(-(b - my[:, :, None])**2 / 2. / sigma2[:, None, None])
        Fy /= (Fy.sum(axis=-1)[:, :, None] + small)
        return Fx, Fy

    def _read(self, x, gamma, Fx, Fy):
        Fyx = (Fy[:, None, :, :, None] * x[:, :, None, :, :]).sum(axis=3)
        FxT = Fx.dimshuffle(0, 2, 1)
        FyxFx = (Fyx[:, :, :, :, None] * FxT[:, None, None, :, :]).sum(axis=3)
        return gamma[:, None, None, None] * FyxFx

    def _get_patch(self, h):
        write_patch = T.dot(h, self.W_patch) + self.b_patch
        write_patch = write_patch.reshape((h.shape[0], self._input_shape[0],
                                           self.N_dec, self.N_dec))
        return write_patch

    def _write(self, write_patch, gamma, Fx, Fy):
        Fyx = (Fy[:, None, :, :, None] * write_patch[:, :, :, None, :]).sum(axis=2)
        FyxFx = (Fyx[:, :, :, :, None] * Fx[:, None, None, :, :]).sum(axis=3)
        return FyxFx / gamma[:, None, None, None]

    def _get_sample(self, h, eps):
        mean = T.dot(h, self.W_mean) + self.b_mean
        # eps = self.theano_rng.normal(avg=0., std=1., size=mean.shape)
        logsigma = T.dot(h, self.W_sigma) + self.b_sigma
        sigma = T.exp(logsigma)
        if self._train_state:
            sample = mean + eps * sigma
        else:
            sample = mean + 0 * eps * sigma
        kl = -.5 - logsigma + .5 * (mean**2 + sigma**2)
        # kl = .5 * (mean**2 + sigma**2 - logsigma - 1)
        return sample, kl.sum(axis=-1)

    def _get_rnn_input(self, x, rnn):
        if self.inner_rnn == 'gru':
            x_z = T.dot(x, rnn.W_z) + rnn.b_z
            x_r = T.dot(x, rnn.W_r) + rnn.b_r
            x_h = T.dot(x, rnn.W_h) + rnn.b_h
            return x_z, x_r, x_h

        elif self.inner_rnn == 'lstm':
            xi = T.dot(x, rnn.W_i) + rnn.b_i
            xf = T.dot(x, rnn.W_f) + rnn.b_f
            xc = T.dot(x, rnn.W_c) + rnn.b_c
            xo = T.dot(x, rnn.W_o) + rnn.b_o
            return xi, xf, xc, xo

    def _get_rnn_state(self, rnn, *args):
        mask = 1.  # no masking
        if self.inner_rnn == 'gru':
            x_z, x_r, x_h, h_tm1 = args
            h = rnn._step(x_z, x_r, x_h, mask, h_tm1,
                          rnn.U_z, rnn.U_r, rnn.U_h)
            return h
        elif self.inner_rnn == 'lstm':
            xi, xf, xc, xo, h_tm1, cell_tm1 = args
            h, cell = rnn._step(xi, xf, xo, xc, mask,
                                h_tm1, cell_tm1,
                                rnn.U_i, rnn.U_f, rnn.U_o, rnn.U_c)
            return h, cell

    def _get_initial_states(self, X):
        batch_size = X.shape[0]
        canvas = self.init_canvas.dimshuffle('x', 0, 1, 2).repeat(batch_size,
                                                                  axis=0)
        init_enc = self.init_h_enc.dimshuffle('x', 0).repeat(batch_size, axis=0)
        init_dec = self.init_h_dec.dimshuffle('x', 0).repeat(batch_size, axis=0)
        if self.inner_rnn == 'lstm':
            init_cell_enc = self.init_cell_enc.dimshuffle('x', 0).repeat(batch_size, axis=0)
            init_cell_dec = self.init_cell_dec.dimshuffle('x', 0).repeat(batch_size, axis=0)
            return canvas, init_enc, init_cell_enc, init_cell_dec
        else:
            return canvas, init_enc, init_dec

    def _step(self, eps, canvas, h_enc, h_dec, x, *args):
        x_hat = x - self.canvas_activation(canvas)
        gx, gy, sigma2, delta, gamma = self._get_attention.trainable_weights(
            h_dec, self.L_enc, self.b_enc, self.N_enc)
        Fx, Fy = self._get_filterbank(gx, gy, sigma2, delta, self.N_enc)
        read_x = self._read(x, gamma, Fx, Fy).flatten(ndim=2)
        read_x_hat = self._read(x_hat, gamma, Fx, Fy).flatten(ndim=2)
        enc_input = T.concatenate([read_x, read_x_hat, h_dec], axis=-1)

        x_enc_z, x_enc_r, x_enc_h = self._get_rnn_input(enc_input, self.enc)
        new_h_enc = self._get_rnn_state(self.enc, x_enc_z, x_enc_r, x_enc_h,
                                        h_enc)
        sample, kl = self._get_sample(new_h_enc, eps)

        x_dec_z, x_dec_r, x_dec_h = self._get_rnn_input(sample, self.dec)
        new_h_dec = self._get_rnn_state(self.dec, x_dec_z, x_dec_r, x_dec_h,
                                        h_dec)

        gx_w, gy_w, sigma2_w, delta_w, gamma_w = self._get_attention.trainable_weights(
            new_h_dec, self.L_dec, self.b_dec, self.N_dec)
        Fx_w, Fy_w = self._get_filterbank(gx_w, gy_w, sigma2_w, delta_w,
                                          self.N_dec)
        write_patch = self._get_patch(new_h_dec)
        new_canvas = canvas + self._write(write_patch, gamma_w, Fx_w, Fy_w)
        return new_canvas, new_h_enc, new_h_dec, kl

    def _step_lstm(self, eps, canvas, h_enc, cell_enc,
                   h_dec, cell_dec, x, *args):
        x_hat = x - self.canvas_activation(canvas)
        gx, gy, sigma2, delta, gamma = self._get_attention.trainable_weights(
            h_dec, self.L_enc, self.b_enc, self.N_enc)
        Fx, Fy = self._get_filterbank(gx, gy, sigma2, delta, self.N_enc)
        read_x = self._read(x, gamma, Fx, Fy).flatten(ndim=2)
        read_x_hat = self._read(x_hat, gamma, Fx, Fy).flatten(ndim=2)
        enc_input = T.concatenate([read_x, read_x_hat, h_dec.flatten(ndim=2)], axis=1)

        x_enc_i, x_enc_f, x_enc_c, x_enc_o = self._get_rnn_input(enc_input,
                                                                 self.enc)
        new_h_enc, new_cell_enc = self._get_rnn_state(
            self.enc, x_enc_i, x_enc_f, x_enc_c, x_enc_o, h_enc, cell_enc)
        sample, kl = self._get_sample(new_h_enc, eps)

        x_dec_i, x_dec_f, x_dec_c, x_dec_o = self._get_rnn_input(sample,
                                                                 self.dec)
        new_h_dec, new_cell_dec = self._get_rnn_state(
            self.dec, x_dec_i, x_dec_f, x_dec_c, x_dec_o, h_dec, cell_dec)

        gx_w, gy_w, sigma2_w, delta_w, gamma_w = self._get_attention.trainable_weights(
            new_h_dec, self.L_dec, self.b_dec, self.N_dec)
        Fx_w, Fy_w = self._get_filterbank(gx_w, gy_w, sigma2_w, delta_w,
                                          self.N_dec)
        write_patch = self._get_patch(new_h_dec)
        new_canvas = canvas + self._write(write_patch, gamma_w, Fx_w, Fy_w)
        return new_canvas, new_h_enc, new_cell_enc, new_h_dec, new_cell_dec, kl

    def get_output(self, train=False):
        self._train_state = train
        X, eps = self.get_input(train).values()
        eps = eps.dimshuffle(1, 0, 2)

        if self.inner_rnn == 'gru':
            outputs, updates = scan(self._step,
                                    sequences=eps,
                                    outputs_info=self._get_initial_states(X) + (None, ),
                                    non_sequences=[X, ] + self.trainable_weights,
                                    # n_steps=self.n_steps,
                                    truncate_gradient=self.truncate_gradient)

        elif self.inner_rnn == 'lstm':
            outputs, updates = scan(self._step_lstm,
                                    sequences=eps,
                                    outputs_info=self._get_initial_states(X) + (None, ),
                                    non_sequences=[X, ] + self.trainable_weights,
                                    truncate_gradient=self.truncate_gradient)

        kl = outputs[-1].sum(axis=0).mean()
        if train:
            # self.updates = updates
            self.regularizers = [SimpleCost(kl), ]
        if self.return_sequences:
            return [outputs[0].dimshuffle(1, 0, 2, 3, 4), kl]
        else:
            return [outputs[0][-1], kl]
