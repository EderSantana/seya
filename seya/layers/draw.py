import theano.tensor as T

from keras.layers.recurrent import GRU, Recurrent


class DRAW(Recurrent):
    '''DRAW

    Parameters:
    ===========
    dim : encoder dimension
    input_shape : (n_channels, rows, cols)
    N : Size of filter bank
    n_steps : number of sampling steps
    '''
    def __init__(self, dim, input_shape, N, n_steps,
                 inner_rnn='gru', truncate_gradient=-1):
        self.dim = dim
        self.input_shape = input_shape
        self.N = N
        self.n_steps = n_steps
        self.truncate_gradient = truncate_gradient

        if len(input_shape) == 2:  # single channel image
            self.height = input_shape[0]
            self.width = input_shape[1]
        else:
            raise ValueError('This image shape is not supported')

        self.inner_rnn = inner_rnn
        if inner_rnn == 'gru':
            self.enc = GRU(input_dim=N**2+dim, output_dim=dim)
            self.dec = GRU(input_dim=dim, output_dim=dim)
        else:
            raise ValueError('This type of rnn is not supported')

        self.L = self.enc.init((dim, 5))  # attention parameters (eq. 21)

    def _get_attention_params(self, h_dec, L):
        p = T.dot(h_dec, L)
        gx = self.width * (p[0]+1) / 2.
        gy = self.height * (p[1]+1) / 2.
        sigma2 = T.exp(p[2])
        delta = T.exp(p[3] * (max(self.width, self.height) - 1) / (self.N - 1))
        gamma = T.exp(p[4])
        return gx, gy, sigma2, delta, gamma

    def _get_filterback(self, gx, gy, sigma2, delta):
        eps = 1e-6
        N = self.N
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
        FyxFx = (Fyx[:, :, :, :, None] * Fx[:, None, None, :, :]).sum(axis=3)
        return gamma * FyxFx

    def _write(self, x, gamma, Fx, Fy):
        pass

    def _sample(self):
        # return sample, kl
        pass

    def _get_rnn_input(self, x, rnn):
        if self.inner_rnn == 'gru':
            x_z = T.dot(x, rnn.W_z) + rnn.b_z
            x_r = T.dot(x, rnn.W_r) + rnn.b_r
            x_h = T.dot(x, rnn.W_h) + rnn.b_h
        return x_z, x_r, x_h

    def _get_rnn_state(self, rnn, *args):
        if self.inner_rnn == 'gru':
            x_z, x_r, x_h, mask, h_tm1 = args
            h = rnn._step(x_z, x_r, x_h, mask, h_tm1,
                          rnn.U_z, rnn.U_r, rnn.U_h)
        return h

    def _step(self, x, mask, canvas, h_enc, h_dec, *args):
        x_hat = x - canvas
        gx, gy, sigma2, delta, gamma = self._get_attention_params(h_dec, self.L)
        Fx, Fy = self._get_filterbank(gx, gy, sigma2, delta)
        read_x = self._read(x, gamma, Fx, Fy).flatten(ndim=2)
        read_x_hat = self._read(x_hat, gamma, Fx, Fy).flatten(ndim=2)
        enc_input = T.concatenate([read_x, read_x_hat], axis=-1)

        x_enc_z, x_enc_r, x_enc_h = self._get_rnn_input(enc_input, self.enc)
        new_h_enc = self._get_rnn_state(self.enc, x_enc_z, x_enc_r, x_enc_h,
                                        mask, h_enc)
        sample, kl = self._get_sample(new_h_enc)

        x_dec_z, x_dec_r, x_dec_h = self._get_rnn_input(sample, self.dec)
        new_h_dec = self._get_rnn_state(self.dec, x_dec_z, x_dec_r, x_dec_h,
                                        mask, h_dec)

        gx_w, gy_w, sigma2_w, delta_w, gamma_w = self._get_attention_params(
            h_dec, self.L_dec)
        Fx_w, Fy_w = self._get_filterback(gx_w, gy_w, sigma2_w, delta_w)
        new_canvas = canvas + self._write(canvas, Fx, Fy)
        return new_canvas, new_h_enc, new_h_dec
