from keras.layers.core import Dense, Activation
from seya.layers.containers import Recursive


model = Recursive()
model.add_input('input', ndim=3)
model.add_state('h', dim=128)
model.add_node(Dense(228, 128), name='rec', inputs=['input', 'h'],
               return_state='h')
model.add_node(Activation('tanh'), name='out', input='rec', create_output=True)
