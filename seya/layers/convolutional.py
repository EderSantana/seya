from keras import backend as K
from keras.layers.core import Layer


class GlobalPooling2D(Layer):
    """
    By @entron
    Borrowed and modified from here: https://github.com/fchollet/keras/pull/522
    """
    def __init__(self, pooling_function='average'):
        super(GlobalPooling2D, self).__init__()
        if pooling_function not in {'average', 'max'}:
            raise Exception('Invalid pooling function for GlobalPooling2D:', pooling_function)
        if pooling_function == 'average':
            self.pooling_function = K.mean
        else:
            self.pooling_function = K.max
        self.input = K.placeholder(ndim=4)

    def get_output(self, train):
        X = self.get_input(train)
        return self.pooling_function(self.pooling_function(X, axis=-1), axis=-1)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "pooling_function": self.pooling_function.__name__}
