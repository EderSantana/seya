from keras import backend as K
from keras.layers.core import Layer, MaskedLayer


class GlobalPooling2D(Layer):
    """
    By @entron
    Borrowed and modified from here: https://github.com/fchollet/keras/pull/522
    """
    def __init__(self, pooling_function='average'):
        super(GlobalPooling2D, self).__init__()
        if pooling_function not in {'average', 'max'}:
            raise Exception('Invalid pooling function for GlobalPooling2D:', pooling_function)
        if pooling_function == 'average':
            self.pooling_function = K.mean
        else:
            self.pooling_function = K.max
        self.input = K.placeholder(ndim=4)

    def get_output(self, train):
        X = self.get_input(train)
        return self.pooling_function(self.pooling_function(X, axis=-1), axis=-1)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "pooling_function": self.pooling_function.__name__}


class ChannelDropout(MaskedLayer):
    def __init__(self, p, **kwargs):
        super(ChannelDropout, self).__init__(**kwargs)
        self.p = p

    def channel_dropout(self, X):
        # TODO write channel_dropout at Keras.backend
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        rng = RandomStreams()
        r = rng.binomial(X.shape[1:2], p=1-self.p, dtype=X.dtype)
        r = r.dimshuffle('x', 0, 'x', 'x').repeat(X.shape[0],
                                                  axis=0).repeat(X.shape[2],
                                                                 axis=2).repeat(X.shape[3],
                                                                                axis=3)
        X = X * r
        return X

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.p > 0.:
            if train:
                X = self.channel_dropout(X)
        return X

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'p': self.p}
        base_config = super(ChannelDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
