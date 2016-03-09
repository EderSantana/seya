from keras import backend as K
from keras.layers.core import Layer, MaskedLayer
from theano import tensor as T  # , scan


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

    @property
    def output_shape(self):
        return self.input_shape[:2]

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


class WinnerTakeAll2D(Layer):
    """Spatial Winner-Take-All
    ref: Winner-Take-All Autoencoders by  Alireza Makhzani, Brendan Frey

    Parameters:
    -----------
    spatial: int, controls spatial sparsity, defines maximum number of non zeros
        in a spatial map
    lifetime, int, controls lifetime sparsity, defines maximum number of non
        zeros in a dimension throughout the batches
    n_largest: int, global sparsity, defines maximum number of non zeros in the
        output tensor.
    previous_mode: bool, flag to use legacy behavior of this layer

    NOTE:
    =====
    This is a Theano only layer

    """
    def __init__(self, n_largest=None,
                 spatial=5, lifetime=5, previous_mode=True, **kwargs):
        if K._BACKEND == "tensorflow" and not previous_mode:
            raise ValueError("This is a Theano-only layer")
        super(WinnerTakeAll2D, self).__init__(**kwargs)
        self.n_largest = n_largest
        self.spatial = spatial
        self.lifetime = lifetime
        self.previous_mode = previous_mode

    def wta_largest(self, c, n=1):
        s = T.sort(c.flatten())
        nl = s[-n]
        c = T.switch(T.ge(c, nl), c, 0.)
        return c

    def wta_lifetime(self, c, n=1):
        s = T.sort(c, axis=0)[-n].dimshuffle('x', 0, 1, 2)
        r = K.switch(T.ge(c, s), c, 0.)
        return r

    def wta_spatial(self, c, n=1):
        c = c.reshape((c.shape[0], c.shape[1], -1))
        s = T.sort(c, axis=2)[:, :, -n].dimshuffle(0, 1, 'x', 'x')
        r = K.switch(T.ge(c, s), c, 0.)
        return r

    def winner_take_all(self, X):
        M = K.max(X, axis=(2, 3), keepdims=True)
        R = K.switch(K.equal(X, M), X, 0.)
        return R

    def get_output(self, train=True):
        X = self.get_input(train)
        if train is False:
            return X
        elif self.previous_mode:
            return self.winner_take_all(X)
        else:
            Y = X
            if self.n_largest:
                Y = self.wta_largest(Y, self.n_largest)
            if self.wta_spatial:
                Y = self.wta_lifetime(Y, self.spatial)
            if self.wta_lifetime:
                Y = self.wta_spatial(Y, self.lifetime)
            return Y

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'n_largest': self.n_largest,
                  'spatial': self.spatial,
                  'lifetime': self.lifetime,
                  'previous_mode': self.previous_mode}
        base_config = super(WinnerTakeAll2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
