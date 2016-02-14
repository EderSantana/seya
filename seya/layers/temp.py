from collections import OrderedDict

from keras import backend as K
from keras.layers.core import Layer
from keras import initializations, activations
from keras.layers.convolutional import MaxPooling1D

from seya.regularizers import GaussianKL, LambdaRegularizer


class VariationalDense(Layer):
    """VariationalDense
        Hidden layer for Variational Autoencoding Bayes method [1].
        This layer projects the input twice to calculate the mean and variance
        of a Gaussian distribution. During training, the output is sampled from
        that distribution as mean + random_noise * variance, during testing the
        output is the mean, i.e the expected value of the encoded distribution.

        Parameters:
        -----------
        batch_size: Both Keras backends need the batch_size to be defined before
            hand for sampling random numbers. Make sure your batch size is kept
            fixed during training. You can use any batch size for testing.

        regularizer_scale: By default the regularization is already proberly
            scaled if you use binary or categorical crossentropy cost functions.
            In most cases this regularizers should be kept fixed at one.

    """
    def __init__(self, output_dim, batch_size, init='glorot_uniform',
                 activation='tanh',
                 weights=None, input_dim=None, regularizer_scale=1, **kwargs):
        self.regularizer_scale = regularizer_scale
        self.batch_size = batch_size
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.initial_weights = weights
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        super(VariationalDense, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.W_mean = self.init((input_dim, self.output_dim))
        self.b_mean = K.zeros((self.output_dim,))
        self.W_logsigma = self.init((input_dim, self.output_dim))
        self.b_logsigma = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W_mean, self.b_mean, self.W_logsigma,
                       self.b_logsigma]

        self.regularizers = []
        reg = self.get_variational_regularization(self.get_input())
        self.regularizers.append(reg)

    def get_variational_regularization(self, X):
        mean = self.activation(K.dot(X, self.W_mean) + self.b_mean)
        logsigma = self.activation(K.dot(X, self.W_logsigma) + self.b_logsigma)
        return GaussianKL(mean, logsigma, regularizer_scale=self.regularizer_scale)

    def get_mean_logsigma(self, X):
        mean = self.activation(K.dot(X, self.W_mean) + self.b_mean)
        logsigma = self.activation(K.dot(X, self.W_logsigma) + self.b_logsigma)
        return mean, logsigma

    def _get_output(self, X, train=False):
        mean, logsigma = self.get_mean_logsigma(X)
        if train:
            eps = K.random_normal((self.batch_size, self.output_dim))
            return mean + K.exp(logsigma) * eps
        else:
            return mean

    def get_output(self, train=False):
        X = self.get_input()
        return self._get_output(X, train)

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)


class SlowSiamese(Layer):
    def __init__(self, encoder, decoder, code_dim, batch_size,
                 beta=0.5, subsample=2, regularizer_scale=0.5,
                 init='glorot_uniform', activation='linear',
                 weights=None, input_dim=None, **kwargs):
        self.regularizer_scale = regularizer_scale
        self.beta = beta
        self.max_pool = MaxPooling1D(subsample)
        self.encoder = encoder
        self.decoder = decoder
        self.variational = VariationalDense(code_dim, batch_size,
                                            input_dim=self.encoder.output_shape[1],
                                            regularizer_scale=regularizer_scale)
        self.batch_size = batch_size
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.code_dim = code_dim
        self.initial_weights = weights
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=4)
        super(SlowSiamese, self).__init__(**kwargs)

    def build(self):
        self.encoder.build()
        self.decoder.build()
        self.variational.build()

        outputs = []
        self.regularizers = []
        input_list = self.get_input()
        if isinstance(input_list, OrderedDict):
            assert len(input_list) == 2
            for X in input_list.values():
                Y = self.encoder(X)
                reg = self.variational.get_variational_regularization(Y)
                self.regularizers.append(reg)
                Y = self.variational._get_output(Y, train=True)
                Y = self.decoder(Y)
                outputs.append(Y)
            pool0 = self.max_pool(K.expand_dims(outputs[0], 2))
            pool1 = self.max_pool(K.expand_dims(outputs[1], 2))
            slow = self.beta * ((pool0 - pool1)**2).mean()
            self.regularizers.append(LambdaRegularizer(slow))
        else:
            Y = self.encoder(input_list)
            reg = self.variational.get_variational_regularization(Y)
            self.regularizers.append(reg)
            Y = self.variational._get_output(Y, train=True)
            Y = self.decoder(Y)

        self.trainable_weights = self.encoder.trainable_weights + self.variational.trainable_weights + self.decoder.trainable_weights

    def get_output(self, train=False):
        input_list = self.get_input()
        outputs = []
        if isinstance(input_list, OrderedDict):
            assert len(input_list) == 2
            for X in input_list.values():
                Y = self.encoder(X)
                Y = self.variational._get_output(Y, train)
                Y = self.decoder(Y)
                outputs.append(Y)
        else:
            Y = self.encoder(input_list)
            Y = self.variational._get_output(Y, train)
            Y = self.decoder(Y)
            outputs.append(Y)
        return outputs

    def get_output_at(self, head, train=False):
        return self.get_output(train)[head]

    def get_output_shape(self, head):
        return self.output_shape

    def encode(self, X):
        Y = self.encoder(X)
        return self.variational(Y)

    def sample(self, X):
        return self.decoder(X)

    @property
    def output_shape(self):
        return (self.input_shape[0], self.decoder.output_dim)
