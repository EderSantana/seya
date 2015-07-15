import types
import theano.tensor as T

from keras.layers.recurrent import Recurrent


def _get_reversed_input(self, train=False):
    if hasattr(self, 'previous'):
        X = self.previous.get_output(train=train)
    else:
        X = self.input
    return X[::-1]


class Bidirectional(Recurrent):
    def __init__(self, forward, backward):
        super(Bidirectional, self).__init__()
        self.forward = forward
        self.backward = backward
        self.params = forward.params + backward.params
        self.input = T.tensor3()
        self.forward.input = self.input
        self.backward.input = self.input

    def set_previous(self, layer):
        if not self.supports_masked_input() and layer.get_output_mask() is not None:
            raise Exception("Attached non-masking layer to layer with masked output")
        self.previous = layer
        self.forward.previous = layer
        self.backward.previous = layer
        self.backward.get_input = types.MethodType(_get_reversed_input, self.backward)

    def get_output(self, train=False):
        Xf = self.forward.get_output(train)
        Xb = self.backward.get_output(train)
        Xb = Xb[::-1]
        return T.concatenate([Xf, Xb], axis=-1)

    def get_config(self):
        new_dict = {}
        for k, v in self.forward.get_cofig.items():
            new_dict['forward_'+k] = v
        for k, v in self.backward.get_cofig.items():
            new_dict['backward_'+k] = v
        new_dict["name"] = self.__class__.__name__
        return new_dict
