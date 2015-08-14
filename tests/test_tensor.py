import numpy as np
from theano import tensor, function
from keras.layers.core import Dense
from seya.layers.tensor import Tensor
from seya.utils import apply_model


def test_tensor():
    h2o = Dense(4, 2)
    model = Tensor(input_dim=3, output_dim=4, causes_dim=2, hid2output=h2o,
                   return_mode='both')
    X = tensor.tensor3()
    Y = apply_model(model, X)
    F = function([X], Y, allow_input_downcast=True)
    x = np.ones((3, 10, 3))
    y = F(x)
    assert y.shape == (3, 10, 4+2)
