from __future__ import division
import numpy as np
import theano
import theano.tensor as T
from theano import scan
floatX = theano.config.floatX

from keras.layers.recurrent import GRU, Recurrent
from keras.utils.theano_utils import shared_zeros

from ..utils import theano_rng
from ..regularizers import SimpleCost
from .attention import SpatialTransformer


class DRAW2(Recurrent):
    '''DRAW

    Parameters:
    ===========
    h_dim : encoder/decoder dimension
    z_dim : random sample dimension (reparametrization trick output)
    input_shape : (n_channels, rows, cols)
    N_enc : Size of the encoder's filter bank (MNIST default: 2)
    N_dec : Size of the decoder's filter bank (MNIST default: 5)
    n_steps : number of sampling steps (or how long it takes to draw, default 64)
    inner_rnn : str with rnn type ('gru' default)
    truncate_gradient : int (-1 default)
    return_sequences : bool (False default)
    '''
    theano_rng = theano_rng()

    def __init__(self, input_shape, h_dim, z_dim, N_enc=2, N_dec=5, n_steps=64,
                 inner_rnn='gru', truncate_gradient=-1, return_sequences=False,
                 canvas_activation=T.nnet.sigmoid):
        # self.input = [T.tensor4(), T.tensor3()]   # should be this, but crashes
                                                    # the compiler if I do it.
        self.h_dim = h_dim  # this is 256 for MNIST
        self.z_dim = z_dim  # this is 100 for MNIST
        self.input_shape = input_shape
        self.N_enc = N_enc
        self.N_dec = N_dec
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.canvas_activation = canvas_activation

        self.height = input_shape[1]
        self.width = input_shape[2]
        if self.height != self.width:
            raise ValueError('Make this easy both of us, use square images for'
                             + ' now.')
        if self.height % N_dec != 0:
            raise ValueError('image size mod N_dec is not zero. I will not be'
                             + ' able to upsample and write the image.')

        # initial weights
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1.
        b[1, 1] = 1.
        b = b.flatten().astype(floatX)

        self.read_factor = input_shape[1] / N_enc
        self.write_factor = N_dec / input_shape[1]
        self._transform = SpatialTransformer._transform

        self.inner_rnn = inner_rnn
        if inner_rnn == 'gru':
            self.enc = GRU(input_dim=2*self.N_enc**2 + h_dim, output_dim=h_dim)
            self.dec = GRU(input_dim=z_dim, output_dim=h_dim)
        else:
            raise ValueError('This type of rnn is not supported')

        self.init_canvas = shared_zeros(input_shape)  # canvas and hidden state
        self.init_h_enc = shared_zeros((h_dim))     # initial values
        self.init_h_dec = shared_zeros((h_dim))     # should be trained
        self.L_enc = shared_zeros((h_dim, 6))  # "read" attention parameters (eq. 21)
        self.L_dec = shared_zeros((h_dim, 6))  # "write" attention parameters (eq. 28)
        self.b_enc = theano.shared(b)  # "read" attention parameters (eq. 21)
        self.b_dec = theano.shared(b)  # "write" attention parameters (eq. 28)
        self.W_patch = self.enc.init((
            h_dim, self.N_dec**2*self.input_shape[0]))
        self.b_patch = shared_zeros(self.N_dec**2*self.input_shape[0])
        self.W_mean = self.enc.init((h_dim, z_dim))
        self.W_sigma = self.enc.init((h_dim, z_dim))
        self.b_mean = shared_zeros((z_dim))
        self.b_sigma = shared_zeros((z_dim))
        self.params = self.enc.params + self.dec.params + [
            self.L_enc, self.L_dec, self.b_enc, self.b_dec, self.W_patch,
            self.b_patch, self.W_mean, self.W_sigma, self.b_mean, self.b_sigma,
            self.init_canvas, self.init_h_enc, self.init_h_dec]

    def init_updates(self):
        self.get_output()  # populate regularizers list

    def _get_attention_params(self, h, L, b):
        theta = T.tanh(T.dot(h, L) + b)
        return theta

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
            sample = mean + 0. * eps
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
        batch_size = X.shape[0]
        canvas = self.init_canvas.dimshuffle('x', 0, 1, 2).repeat(batch_size,
                                                                  axis=0)
        init_enc = self.init_h_enc.dimshuffle('x', 0).repeat(batch_size, axis=0)
        init_dec = self.init_h_dec.dimshuffle('x', 0).repeat(batch_size, axis=0)
        # canvas = alloc_zeros_matrix(*X.shape) + self.init_canvas[None, :, :, :]
        # init_enc = alloc_zeros_matrix(X.shape[0], self.h_dim) + self.init_h_enc[None, :]
        # init_dec = alloc_zeros_matrix(X.shape[0], self.h_dim) + self.init_h_dec[None, :]
        return canvas, init_enc, init_dec

    def _step(self, eps, canvas, h_enc, h_dec, x, *args):
        x_hat = x - self.canvas_activation(canvas)
        theta_read = self._get_attention_params(
            h_dec, self.L_enc, self.b_enc).reshape((x.shape[0], 2, 3))
        read_x = self._transform(theta_read, x, self.read_factor).flatten(ndim=2)
        # read_x = T.cast(read_x, floatX)
        read_x_hat = self._transform(theta_read, x_hat, self.read_factor).flatten(ndim=2)
        # read_x_hat = T.cast(read_x_hat, floatX)
        enc_input = T.concatenate([read_x, read_x_hat, h_dec], axis=-1)

        x_enc_z, x_enc_r, x_enc_h = self._get_rnn_input(enc_input, self.enc)
        new_h_enc = self._get_rnn_state(self.enc, x_enc_z, x_enc_r, x_enc_h,
                                        h_enc)
        sample, kl = self._get_sample(new_h_enc, eps)

        x_dec_z, x_dec_r, x_dec_h = self._get_rnn_input(sample, self.dec)
        new_h_dec = self._get_rnn_state(self.dec, x_dec_z, x_dec_r, x_dec_h,
                                        h_dec)

        theta_write = self._get_attention_params(
            new_h_dec, self.L_dec, self.b_dec).reshape((x.shape[0], 2, 3))
        write_patch = T.tanh(T.dot(new_h_dec, self.W_patch) + self.b_patch)
        write_patch = write_patch.reshape((new_h_dec.shape[0], self.input_shape[0],
                                           self.N_dec, self.N_dec))
        upsample_write = self._transform(theta_write, write_patch, self.write_factor)
        new_canvas = canvas + upsample_write
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
        # self.updates = updates
        if self.return_sequences:
            return outputs[0].dimshuffle(1, 0, 2, 3, 4)
        else:
            return outputs[0][-1]
