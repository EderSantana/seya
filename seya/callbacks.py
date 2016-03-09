import numpy as np
import keras.backend as K
from keras.callbacks import Callback


class ResetRNNState(Callback):
    """
    This is supposed to be used with stateful RNNs like
    seya.layers.recurrent.StatefulGRU

    h: the rnn state
    func: a function that returns true when the state should be reset to zero
    """
    def __init__(self, h, func):
        self.h = h
        self.func = func

    def on_batch_end(self, batch, logs={}):
        if self.func(batch, logs):
            self.h.set_value(self.h.get_value()*0)


class RenormalizeWeight(Callback):
    def __init__(self, W, transpose=False):
        Callback.__init__(self)
        self.W = W
        self.W_shape = K.get_value(self.W).shape
        self.transpose = transpose

    def on_batch_begin(self, batch, logs={}):
        W = self.W.get_value()
        if len(self.W_shape) == 4:
            if self.transpose:
                W = W.transpose(1, 0, 2, 3)
                this_shape = W.shape
            W = W.reshape((self.W_shape[0], -1))
        elif self.transpose:
            W = W.T
        norm = np.sqrt((W**2).sum(axis=-1))
        W /= norm[:, None]
        if self.transpose:
            if len(self.W_shape) == 4:
                W = W.reshape(this_shape)
                W = W.transpose(1, 0, 2, 3)
            else:
                W = W.T
        self.W.set_value(W)
