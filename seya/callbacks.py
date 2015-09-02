import numpy as np
from keras.callbacks import Callback


class RenormalizeWeight(Callback):
    def __init__(self, W, transpose=False):
        Callback.__init__(self)
        self.W = W
        self.W_shape = self.W.get_value().shape
        self.transpose = transpose

    def on_batch_begin(self, batch, logs={}):
        W = self.W.get_value()
        if self.W_shape == 4:
            W = W.reshape((self.W_shape[0], -1))
        if self.transpose:
            W = W.T
        norm = np.sqrt((W**2).sum(axis=-1))
        W /= norm[:, None]
        if self.transpose:
            W = W.T
        W = W.reshape(self.W_shape)
        self.W.set_value(W)
