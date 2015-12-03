import numpy as np

#from keras.models import Sequential
#from seya.layers.base import Replicator


# def test_replicator():
#     ''' Replicator is deprecated, use keras.layers.core.RepeatVector instead'''
#     model = Sequential()
#     model.add(Replicator(10))
#     model.compile(loss="mse", optimizer="sgd")
#
#     x = np.ones((2, 2))
#     y = model.predict(x).astype('float64')
#     np.testing.assert_allclose(y, np.ones((2, 10, 2)))
