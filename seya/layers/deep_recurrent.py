import numpy as np

from keras.layers.core import Dense
from keras.layers.recurrent import Recurrent, LSTM
from keras import initializations, activations
from keras import backend as K

# from theano.printing import Print


class DeepLSTM(Recurrent):
    '''Seq2Seq Deep Long-Short Term Memory unit.
    Inspired byt Sutskever et. al 2014

    This layer outputs ALL the states and cells like [h_0, c_0, ..., h_deeper, c_deeper].
    If you need only the very last states, use a Lambada layer to narrow
    output[:, -2*output_dim:-output_dim]

    Args: similar to regular LSTM
        depth: number of LSTMs to stack
        readout: int, if we should a final Dense layer on top or not. readout is
        this Dense's output_dim
    '''
    def __init__(self, output_dim, depth=1, readout=False, dropout=.5,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', **kwargs):
        self.output_dim = output_dim
        self.depth = depth
        self.readout = readout
        self.dropout = dropout
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self._kwargs = kwargs
        super(DeepLSTM, self).__init__(**kwargs)

    def build(self):
        self.lstms = []
        for i in range(self.depth):
            if i == 0:
                self.lstms.append(LSTM(self.output_dim, self.init, self.inner_init,
                                  self.forget_bias_init, self.activation,
                                  self.inner_activation, **self._kwargs))
            else:
                self._kwargs['input_dim'] = self.output_dim
                self.lstms.append(LSTM(self.output_dim, self.init, self.inner_init,
                                  self.forget_bias_init, self.activation,
                                  self.inner_activation, **self._kwargs))

        [lstm.build() for lstm in self.lstms]

        # Get a flat list of trainable_weights
        self.trainable_weights = [weights for lstm in self.lstms for weights in
                                  lstm.trainable_weights]

        if self.readout:
            self.readout_layer = Dense(self.readout, input_dim=self.output_dim,
                                       activation='softmax')
            self.readout_layer.build()
            self.trainable_weights.extend(self.readout_layer.trainable_weights)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        if self.stateful is not None:
            self.states = []
            for lstm in self.lstms:
                self.states.extend(lstm.states)
        else:
            self.states = [None, None] * self.depth

    def reset_states(self):
        [lstm.reset_states() for lstm in self.lstms]

    def get_initial_states(self, X):
        states = super(DeepLSTM, self).get_initial_states(X)
        if self.readout:
            initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
            reducer = K.zeros((self.input_dim, self.readout))
            initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
            states += [initial_state]
        return states

    def step(self, x, states):
        if self.readout:
            assert len(states) == 2*self.depth+1
            states = states[:-1]
        else:
            assert len(states) == 2*self.depth

        h = []
        # P = Print('[debug] X value: ', attrs=("shape",))
        for i, (h_tm1, c_tm1) in enumerate(zip(states[:-1:2], states[1::2])):
            # x = P(x)
            x, new_states = self.lstms[i].step(x, [h_tm1, c_tm1])
            h.extend(new_states)
            # x = K.dropout(x, self.dropout)  # no dropout on the first layer inputs

        if self.readout:
            h += [self.readout_layer(h[-2])]

        return K.concatenate(h, axis=-1), h

    def dream(self, length=140):
        def _dream_step(x, states):
            # input + states
            assert len(states) == 2*self.depth + 1
            x = states[-1]
            x = K.switch(K.equal(x, K.max(x, axis=-1,
                                          keepdims=True)), 1., 0.)
            states = states[:-1]

            h = []
            for i, (h_tm1, c_tm1) in enumerate(zip(states[:-1:2], states[1::2])):
                x, new_states = self.lstms[i].step(x, [h_tm1, c_tm1])
                h.extend(new_states)

            if self.readout:
                h += [self.readout_layer(h[-2])]
                final = h[-1]
            else:
                h += [h[-2]]
                final = h[-2]

            return final, h

        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # Only the very first time point of the input is used, the others only
        # server to count the lenght of the output sequence
        X = self.get_input(train=False)
        mask = self.get_input_mask(train=False)

        assert K.ndim(X) == 3
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitly the number of timesteps of ' +
                                'your sequences.\n' +
                                'If your first layer is an Embedding, ' +
                                'make sure to pass it an "input_length" ' +
                                'argument. Otherwise, make sure ' +
                                'the first layer has ' +
                                'an "input_shape" or "batch_input_shape" ' +
                                'argument, including the time axis.')
        # if self.stateful:
        #     initial_states = self.states
        # else:
        #     initial_states = self.get_initial_states(X)

        s = self.get_output(train=False)[:, -1]
        idx = [0, ] + list(np.cumsum([self.output_dim]*2*self.depth +
                                     [self.readout, ]))
        initial_states = [s[:, idx[i]:idx[i+1]] for i in range(len(idx)-1)]

        # if self.readout:
        #     initial_states.pop(-1)
        # initial_states.append(X[:, 0])

        last_output, outputs, states = K.rnn(_dream_step, K.zeros((1, length, 1)),
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        return outputs

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "depth": self.depth,
                  "readout": self.readout,
                  "dropout": self.dropout,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.readout:
            return input_shape[:2] + [self.output_dim*2*self.depth +
                                      self.readout]
        else:
            return input_shape[:2] + tuple([self.output_dim*2*self.depth])
