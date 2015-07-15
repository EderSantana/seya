import types
import theano.tensor as T


def _get_reversed_input(self, train=False):
    X = self.get_output(train)
    return X[::-1]


def Bidirectional(Recurrent):
    def __init__(self, forward, backward):
        super(Bidirectional, self).__init__()
        self.forward = forward
        self.backward = backward
        self.params = forward.params + backward.params
        self.regularizers = forward.regularizers + backward.regularizers
        if forward.return_sequence == backward.return_sequence:
            self.return_sequence = forward.return_sequence
        else:
            raise ValueError("Forward and Backward Layers of Biderectional network "
                             "must have the same value for `return_sequence`")
        self.input = T.tensor3()

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
