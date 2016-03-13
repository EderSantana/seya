from keras import backend as K
from keras.layers.core import Layer, Dense
from keras import activations

from seya.layers.embedding import BagEmbedding

import theano.tensor as T


class MemNN(Layer):
    """End-To-End Memory Networks

    # Parameters
        hops : int >= 1
        mode : str in {'untied', 'adjacent', 'rnn'}

    # Reference
        - [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)
    """
    def __init__(self, output_dim, input_dim, input_length, f_nb_words,
                 q_nb_words,
                 hops=1, mode="untied", bow_mode="bow", dropout=0.,
                 mask_zero=False,
                 inner_activation='relu', activation='softmax', **kwargs):

        if mode not in {'untied', 'adjacent', 'rnn'}:
            # TODO implement adjacent and rnn
            raise Exception('Invalid merge mode: ' + str(mode))
            assert mode == 'untied'

        self.input_dim = input_dim
        self.input_length = input_length
        self.f_nb_words = f_nb_words
        self.q_nb_words = q_nb_words
        self.dropout = dropout
        self.hops = hops
        self.mode = mode
        self.bow_mode = bow_mode
        self.mask_zero = mask_zero
        self.output_dim = output_dim
        self.inner_activation = activations.get(inner_activation)
        self.activation = activations.get(activation)
        super(MemNN, self).__init__(**kwargs)

    def build(self):
        # list of embedding layers
        self.question = []
        self.facts = []
        self.memory = []
        self.Ws = []
        self.trainable_weights = []
        for i in range(self.hops):
            q = BagEmbedding(self.input_dim, self.q_nb_words, self.output_dim,
                             1, bow_mode=self.bow_mode,
                             mask_zero=self.mask_zero, dropout=self.dropout)
            q.build()
            f = BagEmbedding(self.input_dim, self.f_nb_words, self.output_dim,
                             self.input_length, bow_mode=self.bow_mode,
                             mask_zero=self.mask_zero, dropout=self.dropout)
            f.build()
            m = BagEmbedding(self.input_dim, self.f_nb_words, self.output_dim,
                             self.input_length, bow_mode=self.bow_mode,
                             mask_zero=self.mask_zero, dropout=self.dropout)
            m.build()
            self.question.append(q)
            self.facts.append(f)
            self.memory.append(m)
            if i == self.hops-1:
                w = Dense(self.output_dim, input_dim=self.output_dim,
                          activation=self.activation)
            else:
                w = Dense(self.output_dim, input_dim=self.output_dim,
                          activation=self.inner_activation)
            w.build()
            self.Ws.append(w)
            for l in (q, f, m, w):
                self.trainable_weights += l.trainable_weights

    def get_input(self, train=False):
        res = []
        # question
        q = K.placeholder(shape=(
            self.input_shape[1][0], 1, self.q_nb_words))
        res.append(q)
        # facts
        f = K.placeholder(shape=(
            self.input_shape[0][0], self.input_length, self.f_nb_words))
        res.append(f)
        return res

    def get_output(self, train=False):
        facts, question = self.get_input()
        # facts = facts[0]
        # question = question[0]
        u = K.sum(question, axis=1)
        for q, f, m, w in zip(self.question, self.facts, self.memory, self.Ws):
            C = f(facts, train=train)
            A = m(facts, train=train)
            B = q(u, train=train)
            match = self._match(A, B)
            att = self._attention(C, match)
            u = w(u + att, train=train)
        return u

    def _attention(self, A, B):
        att = K.permute_dimensions(B, (0, 2, 1))
        att = K.sum(att * B, axis=1)
        return att

    def _match(self, A, B):
        return T.batched_tensordot(A, B, axes=(2, 2))

    @property
    def input_shape(self):
        question_shape = (None, 1, self.q_nb_words)
        facts_shape = (None, self.input_length, self.f_nb_words)
        return [facts_shape, question_shape]

    @property
    def output_shape(self):
        return self.input_shape[0][0], self.output_dim

    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = []
        for q, f, m, w in zip(self.question, self.facts, self.memory, self.Ws):
            for l in (q, f, m, w):
                weights += l.get_weights()
        return weights

    def set_weights(self, weights):
        for q, f, m, w in zip(self.question, self.facts, self.memory, self.Ws):
            for l in (q, f, m):
                l.set_weights(weights[0])
                weights.pop[0]
            w.set_weights(weights[:2])
            weights.pop[0]
            weights.pop[1]

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'inner_activation': self.inner_activation,
                  'activation': self.activation,
                  'input_length': self.input_length,
                  'f_nb_words': self.f_nb_words,
                  'q_nb_words': self.q_nb_words,
                  'dropout': self.dropout,
                  'bow_mode': self.bow_mode,
                  'mask_zero': self.mask_zero,
                  'hops': self.hops,
                  'mode': self.mode}
        base_config = super(MemNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
