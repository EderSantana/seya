import numpy as np
from keras.callbacks import Callback


class RenormalizeWeight(Callback):
    def __init__(self, W):
        Callback.__init__(self)
        self.W = W
        self.W_shape = self.W.get_value().shape

    def on_batch_start(self, batch, logs={}):
        W = self.W.get_value()
        if self.W_shape == 4:
            W = W.reshape((self.W_shape[0], -1))
        norm = np.sqrt((W**2).sum(axis=-1))
        W /= norm[:, None]
        W = W.reshape(self.W_shape)
        self.W.set_value(W)
