import theano.tensor as T
from theano import scan

from keras.layers.recurrent import GRU, Recurrent
from keras.utils.theano_utils import alloc_zeros_matrix, shared_zeros

from ..utils import theano_rng
from ..regularizers import SimpleCost


class DRAW(Recurrent):
    '''DRAW

    Parameters:
    ===========
    dim : encoder/decoder dimension
    input_shape : (n_channels, rows, cols)
    N_enc : Size of the encoder's filter bank
    N_dec : Size of the decoder's filter bank
    n_steps : number of sampling steps (or how long it takes to draw)
    inner_rnn : str with rnn type ('gru' default)
    truncate_gradient : int (-1 default)
    return_sequences : bool (False default)
    '''
    theano_rng = theano_rng()

    def __init__(self, input_shape, dim, N_enc, N_dec,
                 inner_rnn='gru', truncate_gradient=-1, return_sequences=False):
        self.input = T.tensor4()
        self.dim = dim
        self.input_shape = input_shape
        self.N_enc = N_enc
        self.N_dec = N_dec
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.height = input_shape[1]
        self.width = input_shape[2]

        self.inner_rnn = inner_rnn
        if inner_rnn == 'gru':
            self.enc = GRU(input_dim=2*self.N_enc**2 + dim, output_dim=dim)
            self.dec = GRU(input_dim=dim, output_dim=dim)
        else:
            raise ValueError('This type of rnn is not supported')

        self.L_enc = self.enc.init((dim, 5))  # "read" attention parameters (eq. 21)
        self.L_dec = self.enc.init((dim, 5))  # "write" attention parameters (eq. 28)
        self.b_enc = shared_zeros((5))  # "read" attention parameters (eq. 21)
        self.b_dec = shared_zeros((5))  # "write" attention parameters (eq. 28)
        self.W_patch = self.enc.init((dim, self.N_dec**2*self.input_shape[0]))
        self.b_patch = shared_zeros((self.N_dec**2*self.input_shape[0]))
        self.W_mean = self.enc.init((dim, dim))
        self.W_sigma = self.enc.init((dim, dim))
        self.b_mean = self.enc.init((dim, dim))
        self.b_sigma = self.enc.init((dim, dim))
        self.params = self.enc.params + self.dec.params + [
            self.L_enc, self.L_dec, self.b_enc, self.b_dec, self.W_patch,
            self.b_patch, self.W_mean, self.W_sigma, self.b_mean, self.b_sigma]

    def _get_attention_params(self, h, L, b, N):
        p = T.tanh(T.dot(h, L) + b)
        gx = self.width * (p[:, 0]+1) / 2.
        gy = self.height * (p[:, 1]+1) / 2.
        sigma2 = T.exp(p[:, 2])
        delta = T.exp(p[:, 3] * (max(self.width, self.height) - 1) / (N - 1))
        gamma = T.exp(p[:, 4])
        return gx, gy, sigma2, delta, gamma

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
        Fyx = (Fy[:, None, :, :, None] * x[:, :, None, :, :]).sum(axis=3)
        FxT = Fx.dimshuffle(0, 2, 1)
        FyxFx = (Fyx[:, :, :, :, None] * FxT[:, None, None, :, :]).sum(axis=3)
        return gamma[:, None, None, None] * FyxFx

    def _write(self, h, gamma, Fx, Fy):
        write_patch = T.tanh(T.dot(h, self.W_patch) + self.b_patch)
        write_patch = write_patch.reshape((h.shape[0], self.input_shape[0],
                                           self.N_dec, self.N_dec))
        Fyx = (Fy[:, None, :, :, None] * write_patch[:, :, :, None, :]).sum(axis=2)
        FyxFx = (Fyx[:, :, :, :, None] * Fx[:, None, None, :, :]).sum(axis=3)
        return FyxFx / gamma[:, None, None, None]

    def _get_sample(self, h, eps):
        mean = T.tanh(T.dot(h, self.W_mean) + self.b_mean)
        # TODO refactor to get user input instead
        # Solve TODO
        # eps = self.theano_rng.normal(avg=0., std=1., size=mean.shape)
        logsigma = T.tanh(T.dot(h, self.W_sigma) + self.b_sigma)
        sigma = T.exp(logsigma)
        if self._train:
            sample = mean + eps * sigma
        else:
            sample = mean
        kl = -.5 - logsigma + .5 * (mean**2 + sigma**2)
        return sample, kl

    def _get_rnn_input(self, x, rnn):
        if self.inner_rnn == 'gru':
            x_z = T.dot(x, rnn.W_z) + rnn.b_z
            x_r = T.dot(x, rnn.W_r) + rnn.b_r
            x_h = T.dot(x, rnn.W_h) + rnn.b_h
        return x_z, x_r, x_h

    def _get_rnn_state(self, rnn, *args):
        if self.inner_rnn == 'gru':
            x_z, x_r, x_h, h_tm1 = args
            mask = 1.  # no masking
            h = rnn._step(x_z, x_r, x_h, mask, h_tm1,
                          rnn.U_z, rnn.U_r, rnn.U_h)
        return h

    def _get_initial_states(self, X):
        canvas = alloc_zeros_matrix(*X.shape)
        init_enc = alloc_zeros_matrix(X.shape[0], self.dim)
        init_dec = alloc_zeros_matrix(X.shape[0], self.dim)
        return canvas, init_enc, init_dec

    def _step(self, eps, canvas, h_enc, h_dec, x, *args):
        x_hat = x - T.nnet.sigmoid(canvas)
        gx, gy, sigma2, delta, gamma = self._get_attention_params(
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

        gx_w, gy_w, sigma2_w, delta_w, gamma_w = self._get_attention_params(
            h_dec, self.L_dec, self.b_dec, self.N_dec)
        Fx_w, Fy_w = self._get_filterbank(gx_w, gy_w, sigma2_w, delta_w,
                                          self.N_dec)
        new_canvas = canvas + self._write(h_dec, gamma_w, Fx_w, Fy_w)
        return new_canvas, new_h_enc, new_h_dec, kl

    def get_output(self, train=False):
        self._train = train
        X, eps = self.get_input(train)
        eps = eps.dimshuffle(1, 0, 2)
        canvas, init_enc, init_dec = self._get_initial_states(X)

        outputs, updates = scan(self._step,
                                sequences=eps,
                                outputs_info=[canvas, init_enc, init_dec, None],
                                non_sequences=[X] + self.params,
                                truncate_gradient=self.truncate_gradient)
        kl = outputs[-1].mean(axis=(1, 2)).sum()
        self.regularizers = [SimpleCost(kl), ]
        self.updates = updates
        if self.return_sequences:
            return outputs[0].dimshuffle(1, 0, 2, 3, 4)
        else:
            return outputs[0][-1]
