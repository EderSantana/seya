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
        self.question = []
        self.outputs = []
        self.memory = []
        self.Ws = []
        self.trainable_weights = []
        for i in range(self.hops):
            # memory embedding - A
            Wm = self.emb_init((self.input_dim, self.output_dim),
                               name="{}_Wm_{}".format(self.name, i))
            self.memory.append(Wm)
            self.trainable_weights += [Wm]

            # question embedding - B
            Wq = self.emb_init((self.input_dim, self.output_dim),
                               name="{}_Wq_{}".format(self.name, i))
            self.question.append(Wq)
            self.trainable_weights += [Wq]

            # outputs embedding - C
            if self.mode == "adjacent" and i > 1:
                Wo = self.outputs[-1]
            elif self.mode == "untied" or i == 0:
                Wo = self.emb_init((self.input_dim, self.output_dim),
                                   name="{}_Wo_{}".format(self.name, i))
                self.trainable_weights += [Wo]
            self.outputs.append(Wo)

            if self.mode == "adjacent" and i == self.hops-1:
                W = Wm.T
            else:
                W = self.init((self.output_dim, self.input_dim),
                              name="{}_W_{}".format(self.name, i))
                self.trainable_weights += [W]
            b = K.zeros((self.input_dim,),
                        name="{}_b_{}".format(self.name, i))
            self.Ws += [[W, b]]
            self.trainable_weights += [b]

    def get_output(self, train=False):
        inputs = [layer.get_output(train) for layer in self.layers]
        facts, question = inputs
        # WARN make sure input layers are Embedding layers with identity init
        # facts = K.argmax(facts, axis=-1)
        # question = K.argmax(question, axis=-1)
        u = question
        for A, B, C, (W, b) in zip(self.memory, self.question, self.outputs,
                                   self.Ws):
            m = self.lookup(facts, A, self.memory_length)
            q = self.lookup(u, B, 1)  # just 1 question
            c = self.lookup(facts, C, self.memory_length)

            # attention weights
            p = self.attention(m, q)

            # output
            o = self.calc_output(c, p, W, b)
            u = o + u
        return K.softmax(u[:, 0, :])

    def lookup(self, x, W, memory_length):
        # shape: (batch*memory_length, input_length)
        x = K.cast(K.reshape(x, (-1, self.input_length)), 'int32')
        # shape: (batch*memory_length, input_length, output_dim)
        X = K.gather(W, x)
        if self.bow_mode == "bow":
            # shape: (batch*memory_length, output_dim)
            X = K.sum(X, axis=1)
        # shape: (batch, memory_length, output_dim)
        X = K.reshape(X, (-1, memory_length, self.output_dim))
        return X

    def attention(self, m, q):
        # shape: (batch, memory_length, 1)
        p = T.batched_tensordot(m, q, (2, 2))
        # shape: (batch, memory_length)
        p = K.softmax(p[:, :, 0])
        # shape: (batch, 1, memory_length)
        return K.expand_dims(p, dim=1)

    def calc_output(self, c, p, W, b):
        # shape: (batch, memory_length, 1)
        p = K.permute_dimensions(p, (0, 2, 1))
        # shape: (batch, 1, output_dim)
        o = K.sum(c * p, axis=1)
        o = K.dot(o, W) + b
        # shape: (batch, 1, output_dim)
        return K.expand_dims(o, dim=1)
