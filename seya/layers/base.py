import theano.tensor as T
from keras.layers.core import MaskedLayer


class Replicator(MaskedLayer):
    def __init__(self, leng):
        "Replicates an input matrix across a new second dimension"
        super(Replicator, self).__init__()
        self.ones = T.ones((leng,))
        self.input = T.matrix()

    def get_output(self, train=False):
        X = self.get_input(train)
        output = X[:, None, :] * self.ones[None, :, None]
        return output


class TimePicker(MaskedLayer):
    def __init__(self, time=-1):
        '''Picks a single value in time from a recurrent layer
           without forgeting its input mask'''
        super(TimePicker, self).__init__()
        self.time = time
        self.input = T.tensor3()

    def get_output(self, train=False):
        X = self.get_input(train)
        return X[:, self.time, :]
