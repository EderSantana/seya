import theano.tensor as T
import keras.backend as K
from keras.layers.core import LambdaMerge
from keras import initializations


class MemN2N(LambdaMerge):
    def __init__(self, layers, output_dim, input_dim, input_length,
                 memory_length, hops=3, bow_mode="bow", mode="adjacent",
                 emb_init="uniform", init="glorot_uniform", **kwargs):

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.input_length = input_length
        self.memory_length = memory_length
        self.hops = hops
        self.bow_mode = bow_mode
        self.mode = mode
        self.init = initializations.get(init)
        self.emb_init = initializations.get(emb_init)
        output_shape = (self.output_dim, )

        super(MemN2N, self).__init__(layers, lambda x: x, output_shape)

    def build(self):
        # list of embedding layers
        self.outputs = []
        self.memory = []
        # self.Hs = []  # if self.mode == "rnn"
        self.trainable_weights = []
        for i in range(self.hops):
            # memory embedding - A
            if self.mode == "adjacent" and i > 0:
                A = self.outputs[-1]
            else:
                A = self.emb_init((self.input_dim, self.output_dim),
                                  name="{}_A_{}".format(self.name, i))
                self.trainable_weights += [A]
            self.memory.append(A)

            # outputs embedding - C
            # if self.mode == "adjacent" and i > 1:
            #    Wo = self.outputs[-1]
            # elif self.mode == "untied" or i == 0:
            C = self.emb_init((self.input_dim, self.output_dim),
                              name="{}_C_{}".format(self.name, i))
            self.trainable_weights += [C]
            self.outputs.append(C)

            # if self.mode == "rnn"
            # H = self.init((self.output_dim, self.output_dim),
            #               name="{}_H_{}".format(self.name, i))
            # self.trainable_weights += [H]
            # b = K.zeros((self.input_dim,),
            #             name="{}_b_{}".format(self.name, i))
            # self.Hs += [H]
            # self.trainable_weights += [H]

        if self.mode == "adjacent":
            self.W = self.outputs[-1].T
            self.b = K.zeros((self.input_dim,), name="{}_b".format(self.name))
            # self.trainable_weights += [self.b]

        # question embedding - B
        self.B = self.emb_init((self.input_dim, self.output_dim),
                               name="{}_B".format(self.name))
        self.trainable_weights += [self.B]

        # Temporal embedding
        self.Te = self.emb_init((self.input_length, self.output_dim))
        self.trainable_weights += [self.Te]

    def get_output(self, train=False):
        inputs = [layer.get_output(train) for layer in self.layers]
        facts, question = inputs
        # WARN make sure input layers are Embedding layers with identity init
        # facts = K.argmax(facts, axis=-1)
        # question = K.argmax(question, axis=-1)
        u, mask_q = self.lookup(question, self.B, 1)  # just 1 question
        for A, C in zip(self.memory, self.outputs):
            m, mask_m = self.lookup(facts, A, self.memory_length)
            c, mask_c = self.lookup(facts, C, self.memory_length)

            # attention weights
            p = self.attention(m, u, mask_m)

            # output
            o = self.calc_output(c, p)
            u = o + u
        # u = K.dot(u[:, 0, :], self.W) + self.b
        return u[:, 0, :]  # K.softmax(u)

    def lookup(self, x, W, memory_length):
        # shape: (batch*memory_length, input_length)
        x = K.cast(K.reshape(x, (-1, self.input_length)), 'int32')
        mask = K.expand_dims(K.not_equal(x, 0.), dim=-1)
        # shape: (batch*memory_length, input_length, output_dim)
        X = K.gather(W, x)
        if self.bow_mode == "bow":
            # shape: (batch*memory_length, output_dim)
            X = K.sum(X + K.expand_dims(self.Te, 0), axis=1)
        # shape: (batch, memory_length, output_dim)
        X = K.reshape(X, (-1, memory_length, self.output_dim))
        return X, mask

    def attention(self, m, q, mask):
        # mask original shape is (batch*memory_length, input_length, 1)
        # shape (batch, memory)
        mask = K.reshape(mask[:, 0], (-1, self.memory_length))
        # shape: (batch, memory_length, 1)
        p = T.batched_tensordot(m, q, (2, 2))
        # shape: (batch, memory_length)
        p = K.softmax(p[:, :, 0])  # * K.cast(mask, 'float32')
        # shape: (batch, 1, memory_length)
        return K.expand_dims(p, dim=1)

    def calc_output(self, c, p):
        # shape: (batch, memory_length, 1)
        p = K.permute_dimensions(p, (0, 2, 1))
        # shape: (batch, output_dim)
        o = K.sum(c * p, axis=1)
        # if self.mode == "rnn":
        # import theano
        # W = theano.printing.Print('[Debug] W shape: ', attrs=("shape",))(W)
        # o = K.dot(o, W) + b
        # shape: (batch, 1, output_dim)
        return K.expand_dims(o, dim=1)
