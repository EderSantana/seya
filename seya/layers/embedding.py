from __future__ import absolute_import

from keras import backend as K
from keras import initializations, regularizers, constraints
from keras.layers.core import Layer


class BagEmbedding(Layer):
    '''Apply keras.layers.embedding.Embeddinig to a sequence of phrases and
    compresses each phrase into a vector. Output is thus a bag-of-words per
    sequence.
    This layer can only be used as the first layer in a model.
    # Input shape
    3D tensor with shape: `(nb_samples, nb_sequences, sequence_length)`.
    # Output shape
        3D tensor with shape: `(nb_samples, nb_sequences, output_dim)`.
    # Arguments
      input_dim: int >= 0. Size of the vocabulary, ie.
          1 + maximum integer index occurring in the input data.
      nb_words: Maximum number of words per input sequence
      output_dim: int >= 0. Dimension of the dense embedding.
      input_length: Number of sequences in the input.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).
      bow_mode: How to reduce sequence into a `bag of words`.
          modes are "bow" for conventional sum reduce... TODO
      init: name of initialization function for the weights
          of the layer (see: [initializations](../initializations.md)),
          or alternatively, Theano function to use for weights initialization.
          This parameter is only relevant if you don't pass a `weights`
          argument.
      weights: list of numpy arrays to set as initial weights.
          The list should have 1 element, of shape `(input_dim, output_dim)`.
      W_regularizer: instance of the [regularizers](../regularizers.md) module
        (eg. L1 or L2 regularization), applied to the embedding matrix.
      W_constraint: instance of the [constraints](../constraints.md) module
          (eg. maxnorm, nonneg), applied to the embedding matrix.
      mask_zero: Whether or not the input value 0 is a special "padding"
          value that should be masked out.
          This is useful for [recurrent layers](recurrent.md) which may take
          variable length input. If this is `True` then all subsequent layers
          in the model need to support masking or an exception will be raised.
      dropout: float between 0 and 1. Fraction of the embeddings to drop.
    # References
        - [End-To-End Memory Networks](http://arxiv.org/pdf/1503.08895v5.pdf)
    '''
    input_ndim = 3

    def __init__(self, input_dim, nb_words, output_dim, input_length,
                 bow_mode="bow", init='uniform',
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 mask_zero=False,
                 weights=None, dropout=0., **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bow_mode = bow_mode
        self.init = initializations.get(init)
        self.input_length = input_length
        self.nb_words = nb_words
        self.mask_zero = mask_zero
        self.dropout = dropout

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_dim, self.nb_words)
        super(BagEmbedding, self).__init__(**kwargs)

    def build(self):
        self.input = K.placeholder(shape=(
            self.input_shape[0], self.input_length, self.nb_words),
            dtype='int32')
        self.W = self.init((self.input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        # set mask weights to a vector of zeros, doesn't really matter due to
        # masking, but checking that they don't change is useful for debugging.
        W = K.get_value(self.W)
        W[0] = 0. * W[0]
        K.set_value(self.W, W)

        self.trainable_weights = [self.W]
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

    def get_output_mask(self, train=None):
        X = self.get_input(train)
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(X, 0)

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_length, self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        retain_p = 1. - self.dropout
        if train and self.dropout > 0:
            B = K.random_binomial((self.input_dim,), p=retain_p)
        else:
            B = K.ones((self.input_dim)) * retain_p
        # we zero-out rows of W at random
        Xs = K.cast(K.reshape(X, (-1, self.nb_words)), 'int32')

        # (samples*input_length, nb_words, dim)
        out = K.gather(self.W * K.expand_dims(B), Xs)
        out = K.reshape(out, (-1, self.input_length, self.nb_words,
                              self.output_dim))
        # (samples, input_length, nb_words, dim)
        out = out * K.expand_dims(K.not_equal(X, 0), dim=-1)
        if self.bow_mode == "bow":
            out = K.sum(out, axis=2)
        return out

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "nb_words": self.nb_words,
                  "bow_mode": self.bow_mode,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "input_length": self.input_length,
                  "mask_zero": self.mask_zero,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "dropout": self.dropout}
        base_config = super(BagEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
