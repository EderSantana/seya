import theano
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

       if inner_rnn == 'gru':
