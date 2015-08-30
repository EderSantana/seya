import numpy as np

import theano.tensor as T
from theano import function
from keras.layers.core import Dense, Activation
from seya.layers.containers import Recursive


model = Recursive()
model.add_input('input', ndim=3)
model.add_state('h', dim=2)
model.add_node(Dense(4, 2, init='one'), name='rec', inputs=['input', 'h'],
               return_state='h')
model.add_node(Activation('tanh'), name='out', input='rec', create_output=True)

XX = T.matrix()
HH = T.matrix()
A = model._step(XX, HH)
F = function([XX, HH], A)
x = np.ones((2, 2))
h = np.ones((2, 2))
print F(x, h)

X = model.get_input()
Y = model._get_output()
F = function([X], Y, allow_input_downcast=True)

x = np.ones((3, 5, 2))
y = F(x)
print y
